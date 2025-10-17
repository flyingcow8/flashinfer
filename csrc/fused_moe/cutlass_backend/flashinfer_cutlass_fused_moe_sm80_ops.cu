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

#include <array>
#include <sstream>
#include <string>
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

#if !defined(FLASHINFER_ENABLE_TMA_WS)
#define FLASHINFER_ENABLE_TMA_WS 1
#endif

#if !FLASHINFER_ENABLE_TMA_WS
// SM80 builds do not ship the Hopper/Blackwell TMA kernels, so we provide
// stubbed implementations that keep TensorRT-LLM symbols linkable while
// clearly rejecting any runtime use of the TMA-specific entry points.
namespace tensorrt_llm {
std::array<size_t, 17> TmaWarpSpecializedGroupedGemmInput::workspaceBuffers(int num_experts) {
  (void)num_experts;
  return {};
}

size_t TmaWarpSpecializedGroupedGemmInput::workspaceSize(int num_experts) {
  (void)num_experts;
  return 0;
}

void TmaWarpSpecializedGroupedGemmInput::configureWorkspace(int8_t* start_ptr, int num_experts,
                                                            void* gemm_workspace,
                                                            size_t gemm_workspace_size) {
  (void)start_ptr;
  (void)num_experts;
  (void)gemm_workspace;
  (void)gemm_workspace_size;
  C10_THROW_ERROR_FORMATTED(NotImplementedError,
                            "TMA warp specialized grouped GEMM workspace initialization is "
                            "disabled for SM80.");
}

void TmaWarpSpecializedGroupedGemmInput::setFinalizeFusionParams(
    void* final_output, float const* router_scales, int64_t const* expert_first_token_offset,
    int const* source_token_index, void const* bias, int hidden_size, int num_output_tokens) {
  (void)final_output;
  (void)router_scales;
  (void)expert_first_token_offset;
  (void)source_token_index;
  (void)bias;
  (void)hidden_size;
  (void)num_output_tokens;
  C10_THROW_ERROR_FORMATTED(NotImplementedError,
                            "TMA warp specialized grouped GEMM epilogue fusion is disabled for "
                            "SM80.");
}

std::string TmaWarpSpecializedGroupedGemmInput::toString() const {
  return "TMA warp specialized grouped GEMM disabled for SM80.";
}
}  // namespace tensorrt_llm
#endif

namespace torch_ext {

namespace common = tensorrt_llm::common;
namespace kernels = tensorrt_llm::kernels;
using profiler_backend = kernels::GemmProfilerBackend;
using Profile = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

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

    if (use_fp8_block_scaling) {
      C10_THROW_ERROR_FORMATTED(NotImplementedError,
                                "FP8 Block Scaling is not supported on SM80 (Ampere).");
    }
    if (use_w4a8_group_scaling) {
      C10_THROW_ERROR_FORMATTED(NotImplementedError,
                                "W4A8 Group Scaling is not supported on SM80 (Ampere).");
    }

    if (mActivationDtype == c10::ScalarType::Half && mWeightDtype == c10::ScalarType::Half) {
      mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<half, half>>();
    }
#ifdef ENABLE_BF16
    else if (mActivationDtype == c10::ScalarType::BFloat16 &&
             mWeightDtype == c10::ScalarType::BFloat16) {
      mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
    }
#endif
    else {
      C10_THROW_ERROR_FORMATTED(NotImplementedError,
                                "SM80 backend supports FP16 or BF16 activations/weights. "
                                "Activation: "
                                    << torch::toString(mActivationDtype) << ", Weight: "
                                    << torch::toString(mWeightDtype));
    }

    TORCH_CHECK(mKernelRunner,
                "Failed to construct CUTLASS MoE runner for SM80 backend.");
    mGemmTactics = mKernelRunner->getTactics();
    TORCH_CHECK(!mGemmTactics.empty(),
                "No GEMM tactics registered for SM80 fused MoE backend.");
  }

  int64_t get_tactic_num() { return static_cast<int64_t>(mGemmTactics.size()); }

  void run_gemm_profile(torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int64_t,
                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t,
                        int64_t tactic, bool) {
    TORCH_CHECK(mKernelRunner,
                "CUTLASS MoE runner is not initialized for SM80 backend.");
    TORCH_CHECK(mGemmTactics.size() > 0,
                "No GEMM tactics registered for SM80 fused MoE backend.");
    if (tactic >= 0) {
      TORCH_CHECK(static_cast<size_t>(tactic) < mGemmTactics.size(),
                  "Requested SM80 GEMM tactic index is out of range");
    }
  }

  torch::Tensor run_moe(torch::Tensor& output, torch::Tensor const& x,
                        torch::Tensor const& token_selected_experts,
                        torch::Tensor const& token_final_scales,
                        torch::Tensor const& fc1_expert_weights,
                        torch::Tensor const& fc2_expert_weights,
                        torch::List<torch::Tensor> quant_scales,
                        c10::optional<torch::Tensor> input_sf, int64_t tp_size, int64_t tp_rank,
                        int64_t ep_size, int64_t ep_rank, int64_t cluster_size,
                        int64_t cluster_rank, bool min_latency_mode,
                        torch::List<int64_t> gemm_tactics) {
  (void)cluster_size;
  (void)cluster_rank;
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "Output must be a CUDA tensor");
    TORCH_CHECK(!min_latency_mode, "Min latency mode is not supported on SM80 (Ampere).");
    TORCH_CHECK(token_selected_experts.scalar_type() == c10::ScalarType::Int,
                "token_selected_experts must be int32");
    TORCH_CHECK(token_final_scales.scalar_type() == c10::ScalarType::Float,
                "token_final_scales must be float32");
    TORCH_CHECK(quant_scales.size() == 0,
                "INT8/other quantized modes are not enabled for SM80 backend.");

    int64_t num_tokens = x.size(0);
    int64_t hidden_size = x.size(1);
    int64_t inter_size = fc1_expert_weights.size(1) / mInnerDimMultiplier;
    int64_t num_experts = fc1_expert_weights.size(0);
    int64_t top_k = token_selected_experts.size(1);

    TORCH_CHECK(fc1_expert_weights.dim() == 3,
                "fc1_expert_weights must have shape [experts, inter, hidden]");
    TORCH_CHECK(fc2_expert_weights.dim() == 3,
                "fc2_expert_weights must have shape [experts, hidden, inter]");

    auto stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();

    kernels::MOEParallelismConfig parallelism_config(tp_size, tp_rank, ep_size, ep_rank);
    auto activation_type = tensorrt_llm::ActivationType::Swiglu;

    apply_gemm_tactics(gemm_tactics);

    size_t moe_workspace_size = mKernelRunner->getWorkspaceSize(
        num_tokens, hidden_size, inter_size, static_cast<int>(num_experts),
        static_cast<int>(top_k), activation_type, parallelism_config, false,
        mUseFp8BlockScaling, false, false);
    size_t map_bytes = static_cast<size_t>(top_k * num_tokens * sizeof(int));
    std::array<size_t, 2> workspace_sizes{{moe_workspace_size, map_bytes}};
    size_t total_workspace_size = tensorrt_llm::common::calculateTotalWorkspaceSize(
        workspace_sizes.data(), workspace_sizes.size());

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    ensure_workspace(options, total_workspace_size);

    auto* base_ptr = static_cast<int8_t*>(mWorkspace.data_ptr());
    auto* moe_ptr = base_ptr;
    auto* map_ptr = tensorrt_llm::common::nextWorkspacePtr(base_ptr, moe_workspace_size);

    kernels::QuantParams quant_params{};
    kernels::LoraParams lora_params{};
    kernels::MoeMinLatencyParams min_latency_params{};

    mKernelRunner->runMoe(
        x.const_data_ptr(),
        input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr,
        token_selected_experts.data_ptr<int>(),
        token_final_scales.const_data_ptr<float>(),
        fc1_expert_weights.const_data_ptr(),
        nullptr,
        activation_type,
        fc2_expert_weights.const_data_ptr(),
        nullptr,
        quant_params,
        num_tokens,
        hidden_size,
        inter_size,
        static_cast<int>(num_experts),
        static_cast<int>(top_k),
        reinterpret_cast<char*>(moe_ptr),
        output.data_ptr(),
        reinterpret_cast<int*>(map_ptr),
        parallelism_config,
        false,
        lora_params,
        mUseFp8BlockScaling,
        false,
        min_latency_params,
        stream);

    return output;
  }

 private:
  void ensure_workspace(torch::TensorOptions options, size_t bytes) {
    if (!mWorkspace.defined() || static_cast<size_t>(mWorkspace.numel()) < bytes) {
      mWorkspace = torch::empty({static_cast<long>(bytes)}, options);
    }
  }

  Profile pick_tactic(int64_t slot, torch::List<int64_t> const& gemm_tactics) const {
    TORCH_CHECK(!mGemmTactics.empty(),
                "No GEMM tactics registered for SM80 fused MoE backend.");
    int64_t index = (slot < gemm_tactics.size()) ? gemm_tactics[slot] : -1;
    if (index < 0 || index >= static_cast<int64_t>(mGemmTactics.size())) {
      return mGemmTactics.front();
    }
    return mGemmTactics[static_cast<size_t>(index)];
  }

  void apply_gemm_tactics(torch::List<int64_t> const& gemm_tactics) {
    TORCH_CHECK(mKernelRunner,
                "CUTLASS MoE runner is not initialized for SM80 backend.");
    auto gemm1_config = pick_tactic(0, gemm_tactics);
    auto gemm2_config = pick_tactic(1, gemm_tactics);
    mKernelRunner->setTactic(gemm1_config, gemm2_config);
  }

  c10::ScalarType mActivationDtype;
  c10::ScalarType mWeightDtype;
  c10::ScalarType mOutputDtype;
  bool mUseFp8BlockScaling;
  bool mUseW4A8GroupScaling;
  int mInnerDimMultiplier;
  std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> mKernelRunner;
  std::vector<Profile> mGemmTactics;
  torch::Tensor mWorkspace;
};

TORCH_LIBRARY(fused_moe_sm80, m) {
  m.class_<FusedMoeRunner>("FusedMoeRunner")
      .def(torch::init<c10::ScalarType, c10::ScalarType, c10::ScalarType, bool, bool>())
      .def("get_tactic_num", &FusedMoeRunner::get_tactic_num)
      .def("run_gemm_profile", &FusedMoeRunner::run_gemm_profile)
      .def("run_moe", &FusedMoeRunner::run_moe);
}

}  // namespace torch_ext
