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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace flashinfer {
namespace moe_simple {

// Activation types
enum class ActivationType {
  IDENTITY = 0,
  RELU = 1,
  GELU = 2,
  SWIGLU = 3,
};

// Data type size helpers
template <typename T>
struct sizeof_bits {
  static constexpr int value = sizeof(T) * 8;
};

// CUDA error checking
#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t err = call;                                                                        \
    if (err != cudaSuccess) {                                                                      \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));  \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

// Array helpers for vectorized loads/stores
template <typename T, int N>
struct Array {
  T data[N];
  static constexpr int kElements = N;
  using Element = T;

  __host__ __device__ T& operator[](int i) { return data[i]; }
  __host__ __device__ const T& operator[](int i) const { return data[i]; }

  __host__ __device__ void fill(T val) {
#pragma unroll
    for (int i = 0; i < N; i++) {
      data[i] = val;
    }
  }
};

// Type conversion
template <typename InputArray, typename OutputArray>
__host__ __device__ OutputArray arrayConvert(InputArray const& input) {
  using OutType = typename OutputArray::Element;
  static_assert(InputArray::kElements == OutputArray::kElements);
  OutputArray output;
#pragma unroll
  for (int i = 0; i < OutputArray::kElements; i++) {
    output[i] = static_cast<OutType>(input[i]);
  }
  return output;
}

}  // namespace moe_simple
}  // namespace flashinfer
