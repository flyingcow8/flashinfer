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

#include "common.cuh"

namespace flashinfer {
namespace moe_simple {

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

// Finalize MoE routing: unpermute, reduce, and scale expert outputs
// This kernel converts permuted expert outputs back to original token order
// and performs weighted sum across experts
template <typename OutputType, typename GemmOutputType, typename BiasType>
__global__ void finalizeMoeRoutingKernel(
    GemmOutputType const* expanded_permuted_rows,      // GEMM2 output [expanded_num_rows, cols]
    OutputType* reduced_unpermuted_output,             // Final output [num_rows, cols]
    BiasType const* bias,                              // Expert biases [num_experts, cols] or nullptr
    float const* scales,                               // Router scales [num_rows * k]
    int const* expanded_source_row_to_expanded_dest_row,  // Mapping [expanded_num_rows]
    int const* expert_for_source_row,                  // Expert IDs [expanded_num_rows]
    int64_t const orig_cols,                           // Hidden size
    int64_t const experts_per_token) {                 // top_k
  
  int64_t const original_row = blockIdx.x;  // Token index
  int64_t const num_rows = gridDim.x;
  auto const offset = original_row * orig_cols;
  OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;

  // Load 128-bits per thread for efficiency
  constexpr int64_t FINALIZE_ELEM_PER_THREAD =
      128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);

  int64_t const start_offset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

  using BiasElem = Array<BiasType, FINALIZE_ELEM_PER_THREAD>;
  using InputElem = Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = Array<float, FINALIZE_ELEM_PER_THREAD>;

  auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
  auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

  // Process each column element
  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    ComputeElem thread_output;
    thread_output.fill(0.0f);

    // K-way reduction: sum outputs from top-k experts
    for (int64_t k_idx = 0; k_idx < experts_per_token; ++k_idx) {
      // Calculate indices
      int64_t const expanded_original_row = original_row + k_idx * num_rows;
      int64_t const expanded_permuted_row =
          expanded_source_row_to_expanded_dest_row[expanded_original_row];

      // Get scale for this expert
      int64_t const k_offset = original_row * experts_per_token + k_idx;
      float const row_scale = scales ? scales[k_offset] : 1.0f;

      // Load expert output
      auto const* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

      // Get expert ID for bias
      int64_t const expert_idx = expert_for_source_row[k_offset];
      auto const* bias_ptr = bias_v + expert_idx * num_elems_in_col;

      ComputeElem bias_value;
      if (bias) {
        bias_value = arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
      } else {
        bias_value.fill(0.0f);
      }

      // Load, scale, and accumulate
      ComputeElem expert_result =
          arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
      thread_output = thread_output + row_scale * (expert_result + bias_value);
    }

    // Write final result
    OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }
}

// Specialization for summing float arrays
template <int N>
__device__ Array<float, N> operator+(Array<float, N> const& a, Array<float, N> const& b) {
  Array<float, N> result;
#pragma unroll
  for (int i = 0; i < N; i++) {
    result[i] = a[i] + b[i];
  }
  return result;
}

// Specialization for scaling float arrays
template <int N>
__device__ Array<float, N> operator*(float scale, Array<float, N> const& a) {
  Array<float, N> result;
#pragma unroll
  for (int i = 0; i < N; i++) {
    result[i] = scale * a[i];
  }
  return result;
}

// Launcher for finalize MoE routing kernel
template <typename OutputType, typename GemmOutputType, typename BiasType>
void finalizeMoeRoutingKernelLauncher(
    GemmOutputType const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    BiasType const* bias, float const* final_scales,
    int const* expanded_source_row_to_expanded_dest_row, int const* expert_for_source_row,
    int64_t const num_rows, int64_t const cols, int64_t const experts_per_token,
    cudaStream_t stream) {
  
  int64_t const blocks = num_rows;
  int64_t const threads = FINALIZE_THREADS_PER_BLOCK;

  finalizeMoeRoutingKernel<OutputType, GemmOutputType, BiasType><<<blocks, threads, 0, stream>>>(
      expanded_permuted_rows, reduced_unpermuted_output, bias, final_scales,
      expanded_source_row_to_expanded_dest_row, expert_for_source_row, cols, experts_per_token);

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace moe_simple
}  // namespace flashinfer
