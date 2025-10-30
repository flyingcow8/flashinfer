/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "common.cuh"
#include "finalize_kernel.cuh"
#include "routing_kernels.cuh"

namespace flashinfer {
namespace moe_simple {

// GEMM function signature for external implementations
// Users provide their optimized GEMM implementations
using GEMMFunction = std::function<void(
    void* output,         // Output tensor
    void const* input,    // Input tensor
    void const* weights,  // Weight tensor
    void const* bias,     // Bias tensor (can be nullptr)
    int64_t const* expert_first_token_offset,  // Expert token offsets [num_experts + 1]
    int64_t M,            // Number of rows (tokens)
    int64_t N,            // Output dimension
    int64_t K,            // Input dimension
    int num_experts,      // Number of experts
    bool is_gemm1,        // true for GEMM1 (with activation), false for GEMM2
    ActivationType activation,  // Activation function for GEMM1
    cudaStream_t stream)>;

// Main MoE runner class
template <typename T>
class SimpleMoeRunner {
 public:
  SimpleMoeRunner(int num_experts, int hidden_size, int inter_size, int top_k,
                  GEMMFunction gemm1_fn, GEMMFunction gemm2_fn, ActivationType activation)
      : num_experts_(num_experts),
        hidden_size_(hidden_size),
        inter_size_(inter_size),
        top_k_(top_k),
        gemm1_fn_(gemm1_fn),
        gemm2_fn_(gemm2_fn),
        activation_(activation) {}

  void run(void* final_output, void const* input, int const* token_selected_experts,
           float const* token_final_scales, void const* fc1_weights, void const* fc2_weights,
           void const* fc1_biases, void const* fc2_biases, int64_t num_rows,
           cudaStream_t stream);

 private:
  void prepareRouting(int const* token_selected_experts, int64_t num_rows, cudaStream_t stream);

  void permuteInputs(void const* input, float const* token_final_scales, int64_t num_rows,
                     cudaStream_t stream);

  int num_experts_;
  int hidden_size_;
  int inter_size_;
  int top_k_;
  GEMMFunction gemm1_fn_;
  GEMMFunction gemm2_fn_;
  ActivationType activation_;

  // Workspace tensors (allocated on demand)
  thrust::device_vector<int> expanded_dest_row_to_expanded_source_row_;
  thrust::device_vector<int> expanded_source_row_to_expanded_dest_row_;
  thrust::device_vector<int> expert_for_source_row_;
  thrust::device_vector<int64_t> expert_first_token_offset_;
  thrust::device_vector<T> permuted_input_;
  thrust::device_vector<float> permuted_scales_;
  thrust::device_vector<T> intermediate_output_;
  thrust::device_vector<T> gemm2_output_;
};

template <typename T>
void SimpleMoeRunner<T>::prepareRouting(int const* token_selected_experts, int64_t num_rows,
                                        cudaStream_t stream) {
  int64_t expanded_num_rows = num_rows * top_k_;

  // Allocate workspace
  expanded_dest_row_to_expanded_source_row_.resize(expanded_num_rows);
  expanded_source_row_to_expanded_dest_row_.resize(expanded_num_rows);
  expert_for_source_row_.resize(expanded_num_rows);
  expert_first_token_offset_.resize(num_experts_ + 1);

  // Step 1: Flatten token_selected_experts to (expert_id, source_row) pairs
  thrust::device_vector<int> expert_ids(expanded_num_rows);
  thrust::device_vector<int> source_rows(expanded_num_rows);

  // Get raw pointers for device access
  int* expert_ids_ptr = thrust::raw_pointer_cast(expert_ids.data());
  int* source_rows_ptr = thrust::raw_pointer_cast(source_rows.data());
  int* expert_for_source_row_ptr = thrust::raw_pointer_cast(expert_for_source_row_.data());

  // Use thrust to populate arrays
  thrust::counting_iterator<int> iter(0);
  thrust::for_each(thrust::cuda::par.on(stream), iter, iter + expanded_num_rows,
                   [=] __device__(int idx) {
                     int row = idx / top_k_;
                     int k = idx % top_k_;
                     int expert_id = token_selected_experts[row * top_k_ + k];
                     expert_ids_ptr[idx] = expert_id;
                     source_rows_ptr[idx] = row + k * num_rows;  // expanded_source_row
                     expert_for_source_row_ptr[idx] = expert_id;
                   });

  // Step 2: Sort by expert_id to group tokens by expert
  thrust::sort_by_key(thrust::cuda::par.on(stream), expert_ids.begin(), expert_ids.end(),
                      source_rows.begin());

  // Step 3: Build dest -> source mapping
  thrust::copy(thrust::cuda::par.on(stream), source_rows.begin(), source_rows.end(),
               expanded_dest_row_to_expanded_source_row_.begin());

  // Step 4: Calculate expert_first_token_offset
  thrust::fill(thrust::cuda::par.on(stream), expert_first_token_offset_.begin(),
               expert_first_token_offset_.end(), 0);

  // Count tokens per expert
  thrust::device_vector<int> expert_counts(num_experts_, 0);
  thrust::for_each(thrust::cuda::par.on(stream), expert_ids.begin(), expert_ids.end(),
                   [=, counts_ptr = thrust::raw_pointer_cast(expert_counts.data())]
                       __device__(int expert_id) { atomicAdd(&counts_ptr[expert_id], 1); });

  // Compute prefix sum for offsets
  thrust::exclusive_scan(thrust::cuda::par.on(stream), expert_counts.begin(), expert_counts.end(),
                         expert_first_token_offset_.begin());
  expert_first_token_offset_[num_experts_] = expanded_num_rows;
}

template <typename T>
void SimpleMoeRunner<T>::permuteInputs(void const* input, float const* token_final_scales,
                                       int64_t num_rows, cudaStream_t stream) {
  int64_t expanded_num_rows = num_rows * top_k_;

  // Allocate permuted tensors
  permuted_input_.resize(expanded_num_rows * hidden_size_);
  permuted_scales_.resize(expanded_num_rows);

  // Launch expand kernel
  expandInputRowsKernelLauncher<T>(
      static_cast<T const*>(input), thrust::raw_pointer_cast(permuted_input_.data()),
      token_final_scales, thrust::raw_pointer_cast(permuted_scales_.data()),
      thrust::raw_pointer_cast(expanded_dest_row_to_expanded_source_row_.data()),
      thrust::raw_pointer_cast(expanded_source_row_to_expanded_dest_row_.data()), num_rows,
      expanded_num_rows, hidden_size_, top_k_, stream);
}

template <typename T>
void SimpleMoeRunner<T>::run(void* final_output, void const* input,
                             int const* token_selected_experts, float const* token_final_scales,
                             void const* fc1_weights, void const* fc2_weights,
                             void const* fc1_biases, void const* fc2_biases, int64_t num_rows,
                             cudaStream_t stream) {
  int64_t expanded_num_rows = num_rows * top_k_;

  // Step 1: Prepare routing (build mappings)
  prepareRouting(token_selected_experts, num_rows, stream);

  // Step 2: Permute inputs by expert
  permuteInputs(input, token_final_scales, num_rows, stream);

  // Step 3: Allocate intermediate buffers
  intermediate_output_.resize(expanded_num_rows * inter_size_);
  gemm2_output_.resize(expanded_num_rows * hidden_size_);

  // Step 4: Call GEMM1 (input -> intermediate with activation)
  gemm1_fn_(thrust::raw_pointer_cast(intermediate_output_.data()),
            thrust::raw_pointer_cast(permuted_input_.data()), fc1_weights, fc1_biases,
            thrust::raw_pointer_cast(expert_first_token_offset_.data()), expanded_num_rows,
            inter_size_, hidden_size_, num_experts_, true, activation_, stream);

  // Step 5: Call GEMM2 (intermediate -> output)
  gemm2_fn_(thrust::raw_pointer_cast(gemm2_output_.data()),
            thrust::raw_pointer_cast(intermediate_output_.data()), fc2_weights, fc2_biases,
            thrust::raw_pointer_cast(expert_first_token_offset_.data()), expanded_num_rows,
            hidden_size_, inter_size_, num_experts_, false, ActivationType::IDENTITY, stream);

  // Step 6: Finalize (unpermute + reduce + scale)
  finalizeMoeRoutingKernelLauncher<T, T, T>(
      thrust::raw_pointer_cast(gemm2_output_.data()), static_cast<T*>(final_output),
      static_cast<T const*>(fc2_biases), thrust::raw_pointer_cast(permuted_scales_.data()),
      thrust::raw_pointer_cast(expanded_source_row_to_expanded_dest_row_.data()),
      thrust::raw_pointer_cast(expert_for_source_row_.data()), num_rows, hidden_size_, top_k_,
      stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace moe_simple
}  // namespace flashinfer
