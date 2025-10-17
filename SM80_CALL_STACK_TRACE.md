# SM80 MOE 执行流程 - 代码调用栈追踪

## 调用栈深度追踪 (Call Stack Trace)

### 层级 1: Python API 入口

```python
# flashinfer/__init__.py:52
from .fused_moe import cutlass_fused_moe

# 用户代码
from flashinfer import cutlass_fused_moe
output = cutlass_fused_moe(
    input=input_tensor,
    token_selected_experts=token_selected_experts,
    token_final_scales=token_final_scales,
    fc1_expert_weights=fc1_expert_weights,
    fc2_expert_weights=fc2_expert_weights,
    output_dtype=torch.float16,
    quant_scales=[],
)
```

### 层级 2: 主 API 函数 - 架构检测和转发

```python
# flashinfer/fused_moe.py:741-915
def cutlass_fused_moe(...) -> torch.Tensor:
    """主 API，负责架构选择"""
    
    # 步骤 1: 架构检测
    major, minor = get_compute_capability(input.device)  # 返回 (8, x) 对 SM80
    
    # 步骤 2: 架构特定处理
    if major == 8:  # ◄──── SM80 分支
        
        # 步骤 2a: SM80 限制检查
        if use_fp8_block_scaling:
            raise NotImplementedError("FP8 Block Scaling not supported on SM80 (Ampere)")
        if use_w4a8_group_scaling:
            raise NotImplementedError("W4A8 Group Scaling not supported on SM80 (Ampere)")
        if min_latency_mode:
            raise NotImplementedError("Min latency mode not supported on SM80 (Ampere)")
        
        # 步骤 2b: 数据类型检查
        if input.dtype == torch.float8_e4m3fn:
            raise NotImplementedError("FP8 not supported on SM80")
        
        # 步骤 2c: 输出张量分配
        num_rows = input.shape[0]
        hidden_size = fc2_expert_weights.shape[1]
        output_shape = (num_rows, hidden_size)
        
        if output is None:
            output = torch.empty(output_shape, dtype=output_dtype, device=input.device)
        else:
            _check_shape_dtype_device(output, output_shape, output_dtype, ...)
        
        # 步骤 2d: 获取缓存的 SM80 模块并调用
        return get_fused_moe_sm80_module().cutlass_fused_moe_sm80(
            output,
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc2_expert_weights,
            output_dtype,
            quant_scales,
            input_sf,
            tp_size, tp_rank, ep_size, ep_rank,
            cluster_size, cluster_rank,
            use_fp8_block_scaling=False,
            use_w4a8_group_scaling=False,
            min_latency_mode=False,
            tune_max_num_tokens=tune_max_num_tokens,
        )[0]  # ◄──── 返回 list[0] 元素
    
    elif major >= 9:  # ◄──── SM90+ 分支
        # ... SM90+ 特定代码
        pass
```

### 层级 3: 缓存模块加载 - JIT 编译和初始化

```python
# flashinfer/fused_moe.py:340-...)
@functools.cache  # ◄──── 关键: 装饰器缓存结果
def get_fused_moe_sm80_module():
    """
    首次调用:
      1. 调用 gen_fused_moe_sm80_module() 生成 JitSpec
      2. 触发 JIT 编译（~2-5 分钟）
      3. 返回编译后的 C++ 模块
    
    后续调用:
      从缓存返回（~μs）
    """
    
    # 生成 JIT 规范并编译
    module = gen_fused_moe_sm80_module().build_and_load(
        class_name="FusedMoeRunner"
    )
    # 在此处触发编译
    # 编译缓存目录: ~/.cache/flashinfer/fused_moe_sm80/
    
    # 创建包装类
    class MoERunner(TunableRunner):
        _runner_dict: Dict[str, object] = dict()
        
        def __init__(self, x_dtype, weight_dtype, output_dtype, top_k, ...):
            self.x_dtype = x_dtype
            self.weight_dtype = weight_dtype
            self.output_dtype = output_dtype
            # ... 其他初始化
            self._fused_moe_runner = module.FusedMoeRunner(
                c10::ScalarType::Half,     # 对 FP16
                c10::ScalarType::Half,     # 对 FP16
                c10::ScalarType::Half,     # 输出类型
                use_fp8_block_scaling=False,
                use_w4a8_group_scaling=False,
            )
        
        # ... 其他方法
    
    # 返回命名空间
    return SimpleNamespace(
        cutlass_fused_moe_sm80=cutlass_fused_moe_sm80,  # 注册的操作
    )
```

### 层级 4: JIT 规范生成 - 编译单位定义

```python
# flashinfer/fused_moe.py:125-209
def gen_fused_moe_sm80_module() -> JitSpec:
    """
    定义 SM80 MOE 模块的编译规范
    """
    return gen_jit_spec(
        "fused_moe_sm80",  # 缓存键
        sources=[
            # ============ CUTLASS GEMM 核心 ============
            # FP16 实例化
            "csrc/nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_fp16_fp16.cu",
            "csrc/nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_fp16_uint8.cu",
            
            # BF16 实例化
            "csrc/nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_bf16_bf16.cu",
            "csrc/nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/moe_gemm_kernels_bf16_uint8.cu",
            
            # ============ FlashInfer 绑定 ============
            "csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_sm80_ops.cu",
            "csrc/fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu",
            
            # ============ CUTLASS 预生成实例化 ============
            "csrc/nv_internal/tensorrt_llm/cutlass_instantiations/gemm_grouped/cutlass_kernel_file_4.generated.cu",
            "csrc/nv_internal/tensorrt_llm/cutlass_instantiations/gemm_grouped/cutlass_kernel_file_5.generated.cu",
            "csrc/nv_internal/tensorrt_llm/cutlass_instantiations/gemm_grouped/cutlass_kernel_file_6.generated.cu",
            "csrc/nv_internal/tensorrt_llm/cutlass_instantiations/gemm_grouped/cutlass_kernel_file_7.generated.cu",
            
            # ============ 工具库 ============
            "csrc/nv_internal/cpp/common/envUtils.cpp",
            "csrc/nv_internal/cpp/common/logger.cpp",
            "csrc/nv_internal/cpp/common/stringUtils.cpp",
            "csrc/nv_internal/cpp/common/tllmException.cpp",
            "csrc/nv_internal/cpp/common/memoryUtils.cu",
            "csrc/nv_internal/tensorrt_llm/kernels/preQuantScaleKernel.cu",
            "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp",
            "csrc/nv_internal/tensorrt_llm/kernels/lora/lora.cpp",
        ],
        extra_cuda_cflags=[
            "-DFLASHINFER_ENABLE_TMA_WS=0",        # 禁用 TMA（Hopper 特性）
            "-DENABLE_BF16",                       # 启用 BF16
            "-DENABLE_FP16",                       # 启用 FP16
            # 注意: FP8 和 FP4 NOT enabled
            "-DUSING_OSS_CUTLASS_MOE_GEMM",        # 使用开源 CUTLASS
            "-gencode=arch=compute_80,code=sm_80", # SM80 代码生成
        ],
        extra_cflags=[
            "-DFAST_BUILD",  # 优化构建时间
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            "csrc/nv_internal",
            "csrc/nv_internal/include",
            "csrc/nv_internal/tensorrt_llm/cutlass_extensions/include",
            "csrc/nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/include",
            "csrc/nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels",
        ],
    )
```

### 层级 5: 注册的自定义操作 - Python -> C++ 转换

```python
# flashinfer/fused_moe.py:595-735
@register_custom_op(
    "flashinfer::cutlass_fused_moe_sm80",
    mutates_args=(""),
)
def cutlass_fused_moe_sm80(
    output: torch.Tensor,
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    input_sf: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    use_fp8_block_scaling: bool = False,
    use_w4a8_group_scaling: bool = False,
    min_latency_mode: bool = False,
    tune_max_num_tokens: int = 8192,
) -> List[torch.Tensor]:
    """
    这个函数作为 C++ 自定义操作注册
    PyTorch 在调用时自动进行类型转换
    """
    
    # AutoTuner 实例获取
    tuner = AutoTuner.get()
    
    # 动态 token 数量的调优配置
    tune_num_tokens_list = []
    tune_num_tokens = next_positive_power_of_2(tune_max_num_tokens)
    while tune_num_tokens > 0:
        tune_num_tokens_list.append(tune_num_tokens)
        tune_num_tokens //= 2
    
    tuning_config = TuningConfig(
        dynamic_tensors=(
            (0, 0, (tuple(tune_num_tokens_list), next_positive_power_of_2)),
            (3, 0, ((0,), lambda x: x)),
        )
    )
    
    # 创建 MOE 运行器实例
    moe_runner = MoERunner(
        x_dtype=input.dtype,
        weight_dtype=fc1_expert_weights.dtype,
        output_dtype=output_dtype,
        top_k=token_selected_experts.size(1),
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        cluster_size=cluster_size,
        cluster_rank=cluster_rank,
        use_fp8_block_scaling=False,  # SM80 强制禁用
        use_w4a8_group_scaling=False,
    )
    
    # GEMM1 Tactic 自动选择
    _, gemm_tactic_1 = tuner.choose_one(
        "trtllm::fused_moe::gemm1",
        [moe_runner],
        tuning_config,
        [input, fc1_expert_weights, fc2_expert_weights, 
         torch.empty(0)],  # min_latency_tensor
        gemm_idx=1,
    )
    
    # GEMM2 Tactic 自动选择
    _, gemm_tactic_2 = tuner.choose_one(
        "trtllm::fused_moe::gemm2",
        [moe_runner],
        tuning_config,
        [input, fc1_expert_weights, fc2_expert_weights,
         torch.empty(0)],
        gemm_idx=2,
    )
    
    # 执行 C++ run_moe（通过 torch::CustomClassHolder）
    result = moe_runner._fused_moe_runner.run_moe(
        output,
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        quant_scales,
        input_sf,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
        cluster_size,
        cluster_rank,
        min_latency_mode,
        [gemm_tactic_1, gemm_tactic_2],  # ◄──── 关键: 传递调整的 tactics
    )
    
    return [result]
```

### 层级 6: C++ 自定义类 - 主执行逻辑

```cpp
// csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_sm80_ops.cu:97-312

class FusedMoeRunner : public torch::CustomClassHolder {
 public:
  // 构造函数
  FusedMoeRunner(c10::ScalarType activation_dtype,
                 c10::ScalarType weight_dtype,
                 c10::ScalarType output_dtype,
                 bool use_fp8_block_scaling,
                 bool use_w4a8_group_scaling) {
    // 1. 保存数据类型
    mActivationDtype = activation_dtype;      // c10::ScalarType::Half
    mWeightDtype = weight_dtype;              // c10::ScalarType::Half
    mOutputDtype = output_dtype;              // c10::ScalarType::Half
    
    // 2. SM80 限制检查
    if (use_fp8_block_scaling)
        C10_THROW_ERROR("FP8 Block Scaling not supported on SM80");
    if (use_w4a8_group_scaling)
        C10_THROW_ERROR("W4A8 Group Scaling not supported on SM80");
    
    // 3. 创建对应类型的 CUTLASS MOE GEMM 运行器
    if (activation_dtype == c10::ScalarType::Half &&
        weight_dtype == c10::ScalarType::Half) {
        // FP16 路径
        mKernelRunner = std::make_shared<
            kernels::CutlassMoeFCRunner<half, half>>();
    }
#ifdef ENABLE_BF16
    else if (activation_dtype == c10::ScalarType::BFloat16 &&
             weight_dtype == c10::ScalarType::BFloat16) {
        // BF16 路径
        mKernelRunner = std::make_shared<
            kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
    }
#endif
    else {
        C10_THROW_ERROR("SM80 supports only FP16 or BF16");
    }
    
    // 4. 获取该数据类型的所有可用 tactics
    mGemmTactics = mKernelRunner->getTactics();
  }
  
  // 主执行函数
  torch::Tensor run_moe(
      torch::Tensor& output,
      torch::Tensor const& x,
      torch::Tensor const& token_selected_experts,
      torch::Tensor const& token_final_scales,
      torch::Tensor const& fc1_expert_weights,
      torch::Tensor const& fc2_expert_weights,
      torch::List<torch::Tensor> quant_scales,
      c10::optional<torch::Tensor> input_sf,
      int64_t tp_size, int64_t tp_rank,
      int64_t ep_size, int64_t ep_rank,
      int64_t cluster_size, int64_t cluster_rank,
      bool min_latency_mode,
      torch::List<int64_t> gemm_tactics) {  // ◄──── AutoTuner 选择的 tactics
    
    // ============ 验证阶段 ============
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(!min_latency_mode, "Min latency mode not supported on SM80");
    TORCH_CHECK(token_selected_experts.scalar_type() == c10::ScalarType::Int);
    TORCH_CHECK(token_final_scales.scalar_type() == c10::ScalarType::Float);
    TORCH_CHECK(quant_scales.size() == 0,
                "INT8 not enabled for SM80 backend");
    
    // ============ 提取形状 ============
    int64_t num_tokens = x.size(0);
    int64_t hidden_size = x.size(1);
    int64_t inter_size = fc1_expert_weights.size(1) / mInnerDimMultiplier;
    int64_t num_experts = fc1_expert_weights.size(0);
    int64_t top_k = token_selected_experts.size(1);
    
    // ============ 获取 CUDA 流 ============
    auto stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();
    
    // ============ 构造并行化配置 ============
    kernels::MOEParallelismConfig parallelism_config(
        tp_size, tp_rank, ep_size, ep_rank);
    
    // ============ 设置激活函数 ============
    auto activation_type = tensorrt_llm::ActivationType::Swiglu;
    
    // ============ 应用 Auto-Tuner 选择的 Tactics ============
    apply_gemm_tactics(gemm_tactics);  // ◄──── 关键步骤
    
    // ============ 计算工作空间大小 ============
    size_t moe_workspace_size = mKernelRunner->getWorkspaceSize(
        num_tokens, hidden_size, inter_size,
        static_cast<int>(num_experts),
        static_cast<int>(top_k),
        activation_type,
        parallelism_config,
        false,  // use_lora
        mUseFp8BlockScaling,
        false,  // min_latency_mode
        false   // use_awq
    );
    
    size_t map_bytes = static_cast<size_t>(top_k * num_tokens * sizeof(int));
    
    // ============ 分配工作空间 ============
    std::array<size_t, 2> workspace_sizes{{moe_workspace_size, map_bytes}};
    size_t total_workspace_size = 
        tensorrt_llm::common::calculateTotalWorkspaceSize(
            workspace_sizes.data(), workspace_sizes.size());
    
    auto options = torch::TensorOptions()
        .dtype(torch::kInt8)
        .device(x.device());
    ensure_workspace(options, total_workspace_size);
    
    // ============ 工作空间指针计算 ============
    auto* base_ptr = static_cast<int8_t*>(mWorkspace.data_ptr());
    auto* moe_ptr = base_ptr;
    auto* map_ptr = tensorrt_llm::common::nextWorkspacePtr(
        base_ptr, moe_workspace_size);
    
    // ============ 创建参数对象 ============
    kernels::QuantParams quant_params{};      // SM80 使用空配置
    kernels::LoraParams lora_params{};
    kernels::MoeMinLatencyParams min_latency_params{};
    
    // ============ 执行 MOE 运算（关键步骤） ============
    mKernelRunner->runMoe(
        x.const_data_ptr(),
        input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr,
        token_selected_experts.data_ptr<int>(),
        token_final_scales.const_data_ptr<float>(),
        fc1_expert_weights.const_data_ptr(),
        nullptr,  // fc1_expert_biases
        activation_type,
        fc2_expert_weights.const_data_ptr(),
        nullptr,  // fc2_expert_biases
        quant_params,
        num_tokens,
        hidden_size,
        inter_size,
        static_cast<int>(num_experts),
        static_cast<int>(top_k),
        reinterpret_cast<char*>(moe_ptr),  // 工作空间
        output.data_ptr(),                  // 输出缓冲区
        reinterpret_cast<int*>(map_ptr),   // token-expert 映射
        parallelism_config,
        false,  // use_lora
        lora_params,
        mUseFp8BlockScaling,
        false,  // min_latency_mode
        min_latency_params,
        stream   // CUDA 流
    );
    
    return output;
  }

 private:
  void apply_gemm_tactics(torch::List<int64_t> const& gemm_tactics) {
    TORCH_CHECK(mKernelRunner);
    
    // 选择 GEMM1 tactic
    auto gemm1_config = pick_tactic(0, gemm_tactics);
    
    // 选择 GEMM2 tactic
    auto gemm2_config = pick_tactic(1, gemm_tactics);
    
    // 应用到 kernel runner
    mKernelRunner->setTactic(gemm1_config, gemm2_config);
  }
  
  Profile pick_tactic(int64_t slot, torch::List<int64_t> const& gemm_tactics) const {
    if (slot < gemm_tactics.size()) {
      int64_t index = gemm_tactics[slot];
      if (index >= 0 && index < static_cast<int64_t>(mGemmTactics.size())) {
        return mGemmTactics[index];
      }
    }
    return mGemmTactics.front();  // 默认 tactic
  }
};
```

### 层级 7: CUTLASS MOE GEMM 运行器

```cpp
// csrc/nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_kernels.h

template <typename T, typename WeightType, typename OutputType = T, ...>
class CutlassMoeFCRunner : public CutlassMoeFCRunnerInterface {
 public:
  void runMoe(
      void const* input_activations,
      void const* input_sf,
      int const* token_selected_experts,
      float const* token_final_scales,
      void const* fc1_expert_weights,
      void const* fc1_expert_biases,
      ActivationType fc1_activation_type,
      void const* fc2_expert_weights,
      void const* fc2_expert_biases,
      QuantParams quant_params,
      int64_t const num_rows,
      int64_t const hidden_size,
      int64_t const inter_size,
      int const num_experts,
      int const experts_per_token,
      char* workspace_ptr,
      void* final_output,
      int* expanded_source_row_to_expanded_dest_row,
      MOEParallelismConfig parallelism_config,
      bool use_lora,
      LoraParams& lora_params,
      bool use_fp8_block_scaling,
      bool min_latency_mode,
      MoeMinLatencyParams& min_latency_params,
      cudaStream_t stream) override {
    
    // ============ GEMM1 执行 ============
    // 这会调用已在 setTactic() 中选定的 GEMM1 kernel
    this->gemm1(
        input_activations,           // [num_tokens, hidden_size]
        intermediate_result,         // [num_tokens, inter_size]
        fc1_expert_weights,          // [num_experts, inter_size, hidden_size]
        ...
    );
    // ◄──── GPU 上执行 GEMM1 kernel
    // 输入 @ FC1权重ᵀ + SwiGLU激活
    
    // ============ 工作空间同步 ============
    // 等待 GEMM1 完成
    
    // ============ GEMM2 执行 ============
    // 这会调用已在 setTactic() 中选定的 GEMM2 kernel
    this->gemm2(
        intermediate_result,         // [num_tokens, inter_size]
        final_output,                // [num_tokens, hidden_size]
        fc2_expert_weights,          // [num_experts, hidden_size, inter_size]
        token_final_scales,          // [num_tokens, top_k]
        ...
    );
    // ◄──── GPU 上执行 GEMM2 kernel
    // 中间 @ FC2权重ᵀ + Router缩放
  }
};
```

### 层级 8: SM80 Kernel Launcher

```cpp
// csrc/nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl

void sm80_generic_fused_moe_gemm_kernelLauncher(
    ElementType_ const* A,                    // 激活值
    CutlassWeightType_ const* B,              // 权重
    ElementType_ const* biases,
    bool bias_is_broadcast,
    ElementType_* C,                          // 输出
    int64_t const* total_tokens_including_expert,
    int64_t num_rows,
    int64_t gemm_n,
    int64_t gemm_k,
    int num_experts,
    int multi_processor_count,
    cudaStream_t stream,
    int* kernel_occupancy) {
  
  // 定义 kernel 类型（使用 tactic 中的参数）
  constexpr auto activation_type = fused_moe::EpilogueRouting<EpilogueTag>(true);
  using GemmType = fused_moe::Fused_Moe_Kernel_sm80<
      ElementType_,
      CutlassWeightType_,
      ElementType_,
      MaxTileM_,      // ◄──── 来自 tactic
      TileN_,         // ◄──── 来自 tactic
      TileK_,         // ◄──── 来自 tactic
      Stages_,        // ◄──── 来自 tactic
      activation_type
  >;
  
  // 计算占用率
  int occupancy = std::min(2, fused_moe::fused_gemm_maximum_active_blocks<GemmType>());
  int threadblock_count = multi_processor_count * occupancy;
  
  // 创建参数
  using Arguments = typename GemmType::Arguments;
  Arguments args{...};
  auto params = GemmType::to_underlying_arguments(args);
  
  // Launch Kernel
  dim3 grid(params.threadblock_count, 1, 1);
  dim3 block(GemmType::kThreadCount);  // 通常 256-512
  
  fused_moe::run_global<GemmType><<<grid, block, GemmType::kSmemSize, stream>>>(params);
  // ◄──── 关键: Kernel 实际执行
}
```

### 层级 9: CUTLASS Kernel 执行（GPU 上）

```cuda
// 虚拟代码 - 实际 kernel 代码在编译时生成
__global__ void fused_moe_kernel_sm80(
    const half* input,              // [num_tokens, hidden_size]
    const half* weights,            // [num_experts, ...]
    const float* scales,            // [num_tokens, top_k]
    const int* expert_indices,      // [num_tokens, top_k]
    half* output,                   // [num_tokens, hidden_size]
    char* workspace) {
  
  // 1. 线程块初始化
  __shared__ half tile_A[256];      // Shared memory for A tile
  __shared__ half tile_B[256];      // Shared memory for B tile
  int tidx = threadIdx.x;
  int tidx_in_warp = tidx % 32;
  int warp_id = tidx / 32;
  
  // 2. 确定 token 和 expert 分配
  int expert_idx = blockIdx.x % num_experts;
  int token_idx = (blockIdx.x / num_experts) * 128 + tidx_in_warp;
  
  if (token_idx >= num_tokens) return;
  
  // 3. 检查该 token 是否使用该 expert
  bool uses_expert = false;
  for (int k = 0; k < top_k; k++) {
    if (expert_indices[token_idx * top_k + k] == expert_idx) {
      uses_expert = true;
      break;
    }
  }
  
  if (!uses_expert) return;
  
  // 4. GEMM1: 加载并计算
  // 4a. 加载输入 tile
  #pragma unroll
  for (int i = 0; i < ITER; i++) {
    int load_idx = i * 32 + tidx_in_warp;
    tile_A[load_idx] = input[token_idx * hidden_size + load_idx];
  }
  __syncthreads();
  
  // 4b. 加载权重 tile（来自该 expert 的 FC1）
  #pragma unroll
  for (int i = 0; i < ITER; i++) {
    int load_idx = i * 32 + tidx_in_warp;
    tile_B[load_idx] = weights[expert_idx * inter_size * hidden_size + load_idx];
  }
  __syncthreads();
  
  // 4c. 执行 GEMM（使用 Tensor Cores）
  half result = 0;
  #pragma unroll
  for (int k = 0; k < TILES; k++) {
    // Tensor Core GEMM: result += tile_A @ tile_B
    // mma.sync.aligned.m16n8k16.f16.f16 (内在函数)
    result = mma_accumulate(result, tile_A, tile_B);
  }
  
  // 4d. 应用 SwiGLU 激活
  // output = silu(result[0:inter_size/2]) * result[inter_size/2:inter_size]
  half activated = silu(result) * result;
  
  // 5. 写回到工作空间
  workspace[token_idx * inter_size + warp_id] = activated;
  
  // ============ GEMM2 类似流程 ============
  // 从工作空间加载中间结果
  // 乘以 FC2 权重
  // 应用 Router Scale
  // 写回到最终输出
}
```

---

## 数据流图

```
┌──────────────────────────────────────────────────────────────────┐
│ Host Memory (CPU)                                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  input [seq_len, hidden_size]                                   │
│  token_selected_experts [seq_len, top_k]                        │
│  token_final_scales [seq_len, top_k]                            │
│  fc1_expert_weights [num_experts, inter_size, hidden_size]      │
│  fc2_expert_weights [num_experts, hidden_size, inter_size]      │
│                                                                  │
│  ↓ H2D (Host to Device)                                         │
└──────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│ Device Memory (GPU)                                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ GEMM1 Workspace                                    │        │
│  ├─────────────────────────────────────────────────────┤        │
│  │ Input Tile Buffer                [maxM, hidden]   │        │
│  │ Weights Tile Buffer              [inter, hidden]  │        │
│  │ Intermediate Results             [seq, inter]     │        │
│  │ Router Scale Cache               [seq_len, top_k] │        │
│  └─────────────────────────────────────────────────────┘        │
│                    ↓ GEMM1 Kernel                               │
│  Input @ FC1ᵀ + SwiGLU                                         │
│                                                                  │
│  intermediate_cache [seq_len, inter_size]                       │
│                    ↓                                             │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ GEMM2 Workspace                                    │        │
│  ├─────────────────────────────────────────────────────┤        │
│  │ Intermediate Tile Buffer         [maxM, inter]    │        │
│  │ Weights Tile Buffer              [hidden, inter]  │        │
│  │ Output Accumulator               [seq_len, hidden]│        │
│  └─────────────────────────────────────────────────────┘        │
│                    ↓ GEMM2 Kernel                               │
│  Intermediate @ FC2ᵀ + Router Scale                            │
│                                                                  │
│  output [seq_len, hidden_size]                                  │
│                                                                  │
│  ↓ D2H (Device to Host)                                         │
└──────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│ Host Memory (CPU)                                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  result [seq_len, hidden_size]                                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 时间轴（Timeline）

```
事件                          时间       耗时        说明
─────────────────────────────────────────────────────────────────
Python API 调用                T=0      1 μs
  ↓
架构检测                       T=1 μs   1 μs      get_compute_capability()
  ↓
获取 SM80 模块
  ├─ 首次: JIT 编译           T=2 μs   2-5 min  build_and_load()
  └─ 后续: 从缓存加载         T=2 μs   10 μs    @functools.cache
  ↓
参数验证                       T=10 μs  5 μs
  ↓
Auto-Tuner 调整（首次）        T=15 μs  1-2 min  测试不同 tactics
  ↓
创建 MOE 运行器                T=15+ε   10 μs
  ↓
计算工作空间大小              T=25 μs  1 μs
  ↓
分配 GPU 工作空间             T=26 μs  100 μs   H2D 传输
  ↓
应用 GEMM Tactics             T=126 μs 1 μs
  ↓
Kernel Launch (GEMM1)         T=127 μs 1 μs     异步
  ├─ Grid setup               T=127 μs          grid=[occupancy, 1]
  ├─ Block setup              T=128 μs          block=[512]
  └─ Launch                   T=129 μs
  ↓
GEMM1 GPU 执行                T=130 μs 5-10 ms  取决于数据大小
  ├─ Tile 装载                               (缓存命中)
  ├─ GEMM 计算                               (Tensor Core)
  ├─ SwiGLU 激活                             (Epilogue)
  └─ 结果写回                                
  ↓
Kernel Launch (GEMM2)         T=10 ms  1 μs     异步
  ↓
GEMM2 GPU 执行                T=10 ms  5-10 ms
  ├─ 中间装载
  ├─ GEMM 计算
  ├─ Router Scale 应用
  └─ 结果写回
  ↓
D2H 传输（隐式）              T=20 ms  100 μs   结果拷贝回 CPU
  ↓
cudaDeviceSynchronize()       T=20.1ms 1 μs     显式同步
  ↓
返回结果                       T=20.2ms

总计: 20-25 ms （在 NVIDIA A100 上）
```

这个时间线表明:
1. 首次运行需要 JIT 编译 + Auto-Tuning（~3-7 分钟）
2. 后续运行只需要 kernel 执行（~20 ms）
3. AutoTuner 的调优结果缓存在内存中，下次调用复用
