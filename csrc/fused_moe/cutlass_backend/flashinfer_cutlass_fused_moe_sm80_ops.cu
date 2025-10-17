/*
 * Copyright (c) 2025, FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <cstdio>
#include <functional>
#include <iostream>
#include <vector>

#include "moe_gemm_kernels.h"
#include "moe_kernels.h"
#include "pytorch_extension_utils.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#define C10_THROW_ERROR_FORMATTED(ErrorType, ...) \
  do {                                            \
    std::ostringstream oss;                       \
    oss << __VA_ARGS__;                           \
    C10_THROW_ERROR(ErrorType, oss.str());        \
  } while (0)

namespace torch_ext {

namespace common = tensorrt_llm::common;
namespace kernels = tensorrt_llm::kernels;
using profiler_backend = kernels::GemmProfilerBackend;

class FusedMoeRunner : public torch::CustomClassHolder {
 public:
  template <typename Type>
  std::unique_ptr<kernels::CutlassMoeFCRunnerInterface> switch_output_type(
      c10::ScalarType output_type) {
    switch (output_type) {
      case c10::ScalarType::Half:
        return std::make_unique<kernels::CutlassMoeFCRunner<Type, Type, half, Type>>();
#ifdef ENABLE_BF16
      case c10::ScalarType::BFloat16:
        return std::make_unique<kernels::CutlassMoeFCRunner<Type, Type, __nv_bfloat16, Type>>();
#endif
      default:
        C10_THROW_ERROR_FORMATTED(Error, "Invalid output type "
                                             << torch::toString(output_type) << " specified for "
                                             << torch::toString(mActivationDtype));
    }
  };

  FusedMoeRunner(c10::ScalarType activation_dtype, c10::ScalarType weight_dtype,
                 c10::ScalarType output_dtype, bool use_fp8_block_scaling,
                 bool use_w4a8_group_scaling) {
    mActivationDtype = activation_dtype;
    mWeightDtype = weight_dtype;
    mOutputDtype = output_dtype;
    mUseFp8BlockScaling = use_fp8_block_scaling;
    mUseW4A8GroupScaling = use_w4a8_group_scaling;
    mInnerDimMultiplier = 1;

    // SM80 does NOT support FP8 or FP4
    if (use_fp8_block_scaling) {
      C10_THROW_ERROR_FORMATTED(NotImplementedError,
                                "FP8 Block Scaling is not supported on SM80 (Ampere).");
    }
    if (use_w4a8_group_scaling) {
      C10_THROW_ERROR_FORMATTED(NotImplementedError,
                                "W4A8 Group Scaling is not supported on SM80 (Ampere).");
    }

    // SM80 supports: FP16, BF16, FP32
    if (mActivationDtype == c10::ScalarType::Half && mWeightDtype == c10::ScalarType::Half) {
      mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<half, half>>();
    }
#ifdef ENABLE_BF16
    else if (mActivationDtype == c10::ScalarType::BFloat16 &&
             mWeightDtype == c10::ScalarType::BFloat16) {
      mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
    }
#endif
    else if (mActivationDtype == c10::ScalarType::Float &&
             mWeightDtype == c10::ScalarType::Float) {
      mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<float, float>>();
    }
    // Quantized weights support: UINT8, UINT4
    else if (isInt8Quant()) {
      if (mActivationDtype == c10::ScalarType::Half) {
        mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<half, uint8_t>>();
      }
#ifdef ENABLE_BF16
      else if (mActivationDtype == c10::ScalarType::BFloat16) {
        mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, uint8_t>>();
      }
#endif
    } else if (isInt4Quant()) {
      mInnerDimMultiplier = 2;
      if (mActivationDtype == c10::ScalarType::Half) {
        mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<half, cutlass::uint4b_t>>();
      }
#ifdef ENABLE_BF16
      else if (mActivationDtype == c10::ScalarType::BFloat16) {
        mKernelRunner =
            std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>>();
      }
#endif
    } else {
      C10_THROW_ERROR_FORMATTED(NotImplementedError,
                                "Unsupported data type combination for SM80: activation="
                                    << torch::toString(mActivationDtype)
                                    << ", weight=" << torch::toString(mWeightDtype));
    }
  }

  bool isFp8Quant() const {
    return mWeightDtype == c10::ScalarType::Float8_e4m3fn ||
           mWeightDtype == c10::ScalarType::Float8_e5m2;
  }

  bool isInt8Quant() const { return mWeightDtype == c10::ScalarType::Byte; }

  bool isInt4Quant() const {
    return mWeightDtype == c10::ScalarType::UInt4x2;  // Use UInt4x2 for packed UINT4
  }

  bool isNvfp4Quant() const { return mWeightDtype == c10::ScalarType::Long; }  // INT64 == FP4

  int64_t get_tactic_num() { return static_cast<int64_t>(mKernelRunner->getTacticNum()); }

  void run_gemm_profile(torch::Tensor const& x, torch::Tensor const& fc1_expert_weights,
                        torch::Tensor const& fc2_expert_weights, int64_t top_k, int64_t tp_size,
                        int64_t tp_rank, int64_t ep_size, int64_t ep_rank, int64_t cluster_size,
                        int64_t cluster_rank, bool min_latency_mode, int64_t gemm_idx,
                        int64_t tactic, bool do_preparation) {
    // SM80 does not support min_latency_mode
    if (min_latency_mode) {
      C10_THROW_ERROR_FORMATTED(NotImplementedError,
                                "Min latency mode is not supported on SM80 (Ampere).");
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int num_experts = fc1_expert_weights.size(0);
    int hidden_size = x.size(1);
    int inter_size = fc1_expert_weights.size(1) / mInnerDimMultiplier;
    int num_tokens = x.size(0);

    // Profile the GEMM
    if (do_preparation) {
      // Allocate workspace if needed
      size_t ws_size = mKernelRunner->getWorkspaceSize(num_tokens, hidden_size, inter_size,
                                                        num_experts, top_k, kernels::ActivationType::Swiglu);
      if (mWorkspace.size() < ws_size) {
        mWorkspace.resize(ws_size);
      }
    }

    // Run profiling for the specified GEMM (1 or 2)
    if (gemm_idx == 1 || gemm_idx == 2) {
      // Profile this tactic
      try {
        mKernelRunner->gemm_profile(gemm_idx, static_cast<int>(tactic), num_tokens, hidden_size,
                                    inter_size, num_experts, top_k, stream);
      } catch (const std::exception& e) {
        // Profiling failed, skip this tactic
        std::cerr << "Profiling tactic " << tactic << " for GEMM" << gemm_idx
                  << " failed: " << e.what() << std::endl;
      }
    }
  }

  torch::Tensor run_moe(torch::Tensor& output, torch::Tensor const& x,
                        torch::Tensor const& token_selected_experts,
                        torch::Tensor const& token_final_scales,
                        torch::Tensor const& fc1_expert_weights,
                        torch::Tensor const& fc2_expert_weights, torch::List<torch::Tensor> quant_scales,
                        c10::optional<torch::Tensor> input_sf, int64_t tp_size, int64_t tp_rank,
                        int64_t ep_size, int64_t ep_rank, int64_t cluster_size,
                        int64_t cluster_rank, bool min_latency_mode,
                        torch::List<int64_t> gemm_tactics) {
    // SM80 does not support min_latency_mode
    if (min_latency_mode) {
      C10_THROW_ERROR_FORMATTED(NotImplementedError,
                                "Min latency mode is not supported on SM80 (Ampere).");
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int num_experts = fc1_expert_weights.size(0);
    int hidden_size = x.size(1);
    int inter_size = fc1_expert_weights.size(1) / mInnerDimMultiplier;
    int num_tokens = x.size(0);
    int top_k = token_selected_experts.size(1);

    // Allocate workspace
    size_t ws_size = mKernelRunner->getWorkspaceSize(num_tokens, hidden_size, inter_size,
                                                      num_experts, top_k, kernels::ActivationType::Swiglu);
    if (mWorkspace.size() < ws_size) {
      mWorkspace.resize(ws_size);
    }

    // Convert quant_scales to vector (SM80 doesn't use advanced quantization)
    std::vector<torch::Tensor> quant_scales_vec;
    for (size_t i = 0; i < quant_scales.size(); ++i) {
      quant_scales_vec.push_back(quant_scales[i]);
    }

    // Run MoE
    mKernelRunner->runMoe(
        x.data_ptr(), fc1_expert_weights.data_ptr(), fc2_expert_weights.data_ptr(),
        quant_scales_vec, token_selected_experts.data_ptr<int>(),
        token_final_scales.data_ptr(), output.data_ptr(), nullptr,  // no bias
        nullptr,                                                      // no fc2 bias
        num_tokens, hidden_size, inter_size, num_experts, top_k, mWorkspace.data(),
        mWorkspace.size(), stream, gemm_tactics.size() >= 1 ? gemm_tactics[0] : -1,
        gemm_tactics.size() >= 2 ? gemm_tactics[1] : -1);

    return output;
  }

 private:
  c10::ScalarType mActivationDtype;
  c10::ScalarType mWeightDtype;
  c10::ScalarType mOutputDtype;
  bool mUseFp8BlockScaling;
  bool mUseW4A8GroupScaling;
  int mInnerDimMultiplier;
  std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> mKernelRunner;
  common::CudaBuf mWorkspace;
};

TORCH_LIBRARY(fused_moe_sm80, m) {
  m.class_<FusedMoeRunner>("FusedMoeRunner")
      .def(torch::init<c10::ScalarType, c10::ScalarType, c10::ScalarType, bool, bool>())
      .def("get_tactic_num", &FusedMoeRunner::get_tactic_num)
      .def("run_gemm_profile", &FusedMoeRunner::run_gemm_profile)
      .def("run_moe", &FusedMoeRunner::run_moe);
}

}  // namespace torch_ext
