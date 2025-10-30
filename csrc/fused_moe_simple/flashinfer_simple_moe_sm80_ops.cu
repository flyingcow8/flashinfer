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

#include <torch/extension.h>
#include <torch/custom_class.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "moe_runner.cuh"

namespace flashinfer {
namespace moe_simple {

// Forward declarations for GEMM functions that users must provide
// These are weak symbols that can be overridden
extern "C" __attribute__((weak)) void simple_moe_gemm1(
    void* output, void const* input, void const* weights, void const* bias,
    int64_t const* expert_offsets, int64_t M, int64_t N, int64_t K,
    int num_experts, bool is_fp16, cudaStream_t stream);

extern "C" __attribute__((weak)) void simple_moe_gemm2(
    void* output, void const* input, void const* weights, void const* bias,
    int64_t const* expert_offsets, int64_t M, int64_t N, int64_t K,
    int num_experts, bool is_fp16, cudaStream_t stream);

// Stub GEMM implementations that will be used if user doesn't provide custom ones
// These will fail at runtime with clear error messages
void stub_gemm1(void* output, void const* input, void const* weights, void const* bias,
                int64_t const* expert_offsets, int64_t M, int64_t N, int64_t K,
                int num_experts, bool is_gemm1, ActivationType activation,
                cudaStream_t stream) {
  throw std::runtime_error(
      "GEMM1 implementation not provided! "
      "Please implement simple_moe_gemm1() or provide gemm1_fn callback.");
}

void stub_gemm2(void* output, void const* input, void const* weights, void const* bias,
                int64_t const* expert_offsets, int64_t M, int64_t N, int64_t K,
                int num_experts, bool is_gemm1, ActivationType activation,
                cudaStream_t stream) {
  throw std::runtime_error(
      "GEMM2 implementation not provided! "
      "Please implement simple_moe_gemm2() or provide gemm2_fn callback.");
}

// PyTorch class wrapper - simplified for SM80
class FusedMoeRunner : public torch::CustomClassHolder {
 public:
  FusedMoeRunner(at::ScalarType x_dtype, at::ScalarType weight_dtype,
                 at::ScalarType output_dtype)
      : x_dtype_(x_dtype),
        weight_dtype_(weight_dtype),
        output_dtype_(output_dtype) {
    // Validate supported data types
    if (x_dtype != at::kFloat16 && x_dtype != at::kBFloat16 && x_dtype != at::kInt8) {
      throw std::runtime_error(
          "SM80 simplified MoE only supports FP16, BF16, and INT8 input types");
    }
  }

  // Get number of tactics for autotuner (always 1 for simplified implementation)
  int64_t get_tactic_num() const {
    return 1;
  }

  // Profile GEMM for autotuner (no-op for simplified implementation)
  void run_gemm_profile(at::Tensor x, at::Tensor fc1_expert_weights,
                        at::Tensor fc2_expert_weights, int64_t top_k) {
    // No-op: simplified implementation doesn't need profiling
  }

  // Main MoE execution function (simplified signature)
  at::Tensor run_moe(at::Tensor output, at::Tensor input,
                     at::Tensor token_selected_experts,
                     at::Tensor token_final_scales, at::Tensor fc1_expert_weights,
                     at::Tensor fc2_expert_weights,
                     std::vector<at::Tensor> quant_scales) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D [seq_len, hidden_size]");
    TORCH_CHECK(token_selected_experts.dtype() == at::kInt32,
                "token_selected_experts must be int32");
    TORCH_CHECK(token_final_scales.dtype() == at::kFloat32,
                "token_final_scales must be float32")

    // Extract dimensions
    int64_t num_rows = input.size(0);
    int64_t hidden_size = input.size(1);
    int64_t top_k = token_selected_experts.size(1);
    int64_t num_experts = fc1_expert_weights.size(0);
    int64_t inter_size = fc1_expert_weights.size(1);

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const at::cuda::CUDAGuard device_guard(input.device());

    // Determine activation type (default to Swiglu for compatibility)
    ActivationType activation = ActivationType::SWIGLU;

    // Create GEMM function wrappers
    auto gemm1_fn = [](void* output, void const* input, void const* weights, void const* bias,
                       int64_t const* expert_offsets, int64_t M, int64_t N, int64_t K,
                       int num_experts, bool is_gemm1, ActivationType activation,
                       cudaStream_t stream) {
      if (simple_moe_gemm1) {
        // User provided implementation
        bool is_fp16 = true;  // Will be templated
        simple_moe_gemm1(output, input, weights, bias, expert_offsets, M, N, K, num_experts,
                         is_fp16, stream);
      } else {
        stub_gemm1(output, input, weights, bias, expert_offsets, M, N, K, num_experts, is_gemm1,
                   activation, stream);
      }
    };

    auto gemm2_fn = [](void* output, void const* input, void const* weights, void const* bias,
                       int64_t const* expert_offsets, int64_t M, int64_t N, int64_t K,
                       int num_experts, bool is_gemm1, ActivationType activation,
                       cudaStream_t stream) {
      if (simple_moe_gemm2) {
        bool is_fp16 = true;  // Will be templated
        simple_moe_gemm2(output, input, weights, bias, expert_offsets, M, N, K, num_experts,
                         is_fp16, stream);
      } else {
        stub_gemm2(output, input, weights, bias, expert_offsets, M, N, K, num_experts, is_gemm1,
                   activation, stream);
      }
    };

    // Run MoE based on data type
    if (input.dtype() == at::kFloat16) {
      SimpleMoeRunner<half> runner(num_experts, hidden_size, inter_size, top_k, gemm1_fn, gemm2_fn,
                                    activation);
      runner.run(output.data_ptr(), input.data_ptr<half>(),
                 token_selected_experts.data_ptr<int32_t>(), token_final_scales.data_ptr<float>(),
                 fc1_expert_weights.data_ptr<half>(), fc2_expert_weights.data_ptr<half>(), nullptr,
                 nullptr, num_rows, stream);
    } else if (input.dtype() == at::kBFloat16) {
      SimpleMoeRunner<__nv_bfloat16> runner(num_experts, hidden_size, inter_size, top_k, gemm1_fn,
                                             gemm2_fn, activation);
      runner.run(output.data_ptr(), input.data_ptr<__nv_bfloat16>(),
                 token_selected_experts.data_ptr<int32_t>(), token_final_scales.data_ptr<float>(),
                 fc1_expert_weights.data_ptr<__nv_bfloat16>(),
                 fc2_expert_weights.data_ptr<__nv_bfloat16>(), nullptr, nullptr, num_rows, stream);
    } else if (input.dtype() == at::kInt8) {
      // INT8 quantized path - need to handle dequantization
      TORCH_CHECK(quant_scales.size() >= 4, "Need at least 4 quant scales for INT8");
      
      // For simplicity, we'll cast INT8 to FP16 in the runner
      // A production implementation would handle quantization properly
      SimpleMoeRunner<int8_t> runner(num_experts, hidden_size, inter_size, top_k, gemm1_fn,
                                      gemm2_fn, activation);
      runner.run(output.data_ptr(), input.data_ptr<int8_t>(),
                 token_selected_experts.data_ptr<int32_t>(), token_final_scales.data_ptr<float>(),
                 fc1_expert_weights.data_ptr<int8_t>(), fc2_expert_weights.data_ptr<int8_t>(),
                 nullptr, nullptr, num_rows, stream);
    } else {
      throw std::runtime_error("Unsupported input dtype for simple MoE SM80");
    }

    return output;
  }

 private:
  at::ScalarType x_dtype_;
  at::ScalarType weight_dtype_;
  at::ScalarType output_dtype_;
};

}  // namespace moe_simple
}  // namespace flashinfer

// PyTorch library fragment definition (consistent with other FlashInfer ops)
TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.class_<flashinfer::moe_simple::FusedMoeRunner>("FusedMoeRunner")
      .def(torch::init<at::ScalarType, at::ScalarType, at::ScalarType>())
      .def("get_tactic_num", &flashinfer::moe_simple::FusedMoeRunner::get_tactic_num)
      .def("run_gemm_profile", &flashinfer::moe_simple::FusedMoeRunner::run_gemm_profile)
      .def("run_moe", &flashinfer::moe_simple::FusedMoeRunner::run_moe);
}
