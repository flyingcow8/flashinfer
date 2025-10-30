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

constexpr static int EXPAND_THREADS_PER_BLOCK = 256;

// Expand and permute input rows for MoE routing
// This kernel duplicates input tokens according to expert selection and reorders them by expert
template <typename T>
__global__ void expandInputRowsKernel(T const* unpermuted_input, T* permuted_output,
                                      float const* unpermuted_scales, float* permuted_scales,
                                      int const* expanded_dest_row_to_expanded_source_row,
                                      int* expanded_source_row_to_expanded_dest_row,
                                      int64_t const num_rows, int64_t const cols, int64_t const k) {
  int64_t const expanded_dest_row = blockIdx.x;
  int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];

  // Build reverse mapping (source -> dest) for finalize stage
  if (threadIdx.x == 0) {
    expanded_source_row_to_expanded_dest_row[expanded_source_row] =
        static_cast<int>(expanded_dest_row);
  }

  // Load 128-bits per thread for efficient memory access
  constexpr int64_t ELEM_PER_THREAD = 128 / sizeof_bits<T>::value;
  using DataElem = Array<T, ELEM_PER_THREAD>;

  // Calculate source row in original input
  int64_t const source_row = expanded_source_row % num_rows;
  int64_t const source_k_rank = expanded_source_row / num_rows;

  auto const* source_row_ptr = reinterpret_cast<DataElem const*>(unpermuted_input + source_row * cols);
  auto* dest_row_ptr = reinterpret_cast<DataElem*>(permuted_output + expanded_dest_row * cols);

  int64_t const start_offset = threadIdx.x;
  int64_t const stride = EXPAND_THREADS_PER_BLOCK;
  int64_t const num_elems_in_col = cols / ELEM_PER_THREAD;

  // Copy data with vectorized loads/stores
  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    dest_row_ptr[elem_index] = source_row_ptr[elem_index];
  }

  // Copy scales if provided
  if (permuted_scales && threadIdx.x == 0) {
    int64_t const source_k_idx = source_row * k + source_k_rank;
    permuted_scales[expanded_dest_row] =
        unpermuted_scales ? unpermuted_scales[source_k_idx] : 1.0f;
  }
}

// Launcher for expand input rows kernel
template <typename T>
void expandInputRowsKernelLauncher(T const* unpermuted_input, T* permuted_output,
                                   float const* unpermuted_scales, float* permuted_scales,
                                   int const* expanded_dest_row_to_expanded_source_row,
                                   int* expanded_source_row_to_expanded_dest_row,
                                   int64_t const num_rows, int64_t const expanded_num_rows,
                                   int64_t const cols, int const k, cudaStream_t stream) {
  int64_t const blocks = expanded_num_rows;
  int64_t const threads = EXPAND_THREADS_PER_BLOCK;

  expandInputRowsKernel<T><<<blocks, threads, 0, stream>>>(
      unpermuted_input, permuted_output, unpermuted_scales, permuted_scales,
      expanded_dest_row_to_expanded_source_row, expanded_source_row_to_expanded_dest_row, num_rows,
      cols, k);

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace moe_simple
}  // namespace flashinfer
