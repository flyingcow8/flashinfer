# SM80 CUTLASS Fused MOE API 执行流程分析

## 概述

本文档详细分析 FlashInfer 中 `cutlass_fused_moe` API 在 SM80 (Ampere) 架构上的完整执行流程，包括 Host 侧 Python 调用、JIT 编译、Kernel Launch 以及 Device 侧 GPU 执行。

---

## 第一部分：整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Python User Code                           │
│  from flashinfer import cutlass_fused_moe                      │
│  output = cutlass_fused_moe(...)                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         Host Side: PyTorch Custom Operations                   │
│   flashinfer/fused_moe.py:cutlass_fused_moe()                 │
│   - 参数验证与转发                                              │
│   - SM架构检测 (get_compute_capability)                         │
│   - 调用 get_fused_moe_sm80_module()                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│    JIT 编译: flashinfer/fused_moe.py:gen_fused_moe_sm80_module │
│   - 生成 JitSpec，指定编译源文件                                │
│   - 编译标志: DFLASHINFER_ENABLE_TMA_WS=0, -gencode=sm_80      │
│   - 编译缓存在 ~/.cache/flashinfer/                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│   CustomClass: FusedMoeRunner (C++ Class)                      │
│   csrc/fused_moe/cutlass_backend/flashinfer_..._sm80_ops.cu   │
│   - 初始化 CutlassMoeFCRunner<half, half> 或 <bf16, bf16>     │
│   - 管理 GEMM tactics                                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│     Kernel Launcher & Workspace Management                     │
│   csrc/nv_internal/tensorrt_llm/kernels/.../                  │
│   - 分配工作空间 (workspace)                                    │
│   - 计算 GEMM 配置                                              │
│   - 选择 Tactic                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Device Side: CUTLASS Kernels (SM80 Specialized)              │
│   moe_gemm_kernels_*.cu, fused_moe_kernel.cuh                 │
│   - GEMM1: Input -> Intermediate (with SwiGLU activation)     │
│   - GEMM2: Intermediate -> Output (with scaling)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第二部分：Host 侧详细执行流程

### 2.1 Python API 入口点

**文件**: `flashinfer/fused_moe.py` (Line 741-915)

```python
def cutlass_fused_moe(
    input: torch.Tensor,                    # [seq_len, hidden_size]
    token_selected_experts: torch.Tensor,   # [seq_len, top_k]
    token_final_scales: torch.Tensor,       # [seq_len, top_k]
    fc1_expert_weights: torch.Tensor,       # [num_experts, inter_size, hidden_size]
    fc2_expert_weights: torch.Tensor,       # [num_experts, hidden_size, inter_size]
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],       # 量化参数（SM80不使用）
    input_sf: Optional[torch.Tensor] = None,
    # ... 并行化参数 ...
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
```

**执行步骤**:

1. **架构检测** (Line 825-830):
   ```python
   major, minor = get_compute_capability(input.device)
   
   if major == 8:  # SM80, SM86, SM87, SM89
       # SM80 特定处理
   elif major >= 9:  # SM90+ (Hopper/Blackwell)
       # SM90+ 特定处理
   ```

2. **SM80 特定验证** (Line 831-857):
   ```python
   if use_fp8_block_scaling:
       raise NotImplementedError("FP8 Block Scaling not supported on SM80")
   if use_w4a8_group_scaling:
       raise NotImplementedError("W4A8 Group Scaling not supported on SM80")
   if min_latency_mode:
       raise NotImplementedError("Min latency mode not supported on SM80")
   ```
   
   **为什么 SM80 不支持这些特性**:
   - **FP8 Block Scaling**: Ampere (SM80) 的 FP8 支持有限，需要 Hopper (SM90) 的 TMA 指令
   - **W4A8 Group Scaling**: 需要更高级的量化指令集
   - **Min Latency Mode**: 需要 Hopper 的 Warp Specialized TMA kernels

3. **输出张量分配** (Line 858-872):
   ```python
   num_rows = input.shape[0]
   hidden_size = fc2_expert_weights.shape[1]
   output_shape = (num_rows, hidden_size)
   
   if output is None:
       output = torch.empty(output_shape, dtype=output_dtype, device=input.device)
   ```

4. **转发到 SM80 模块** (Line 873-890):
   ```python
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
       use_fp8_block_scaling,
       use_w4a8_group_scaling,
       min_latency_mode,
       tune_max_num_tokens,
   )[0]
   ```

### 2.2 JIT 模块生成

**文件**: `flashinfer/fused_moe.py` (Line 125-209)

```python
def gen_fused_moe_sm80_module() -> JitSpec:
    return gen_jit_spec(
        "fused_moe_sm80",  # 模块名称（用于缓存键）
        [
            # CUTLASS GEMM 核心实现
            "nv_internal/.../moe_gemm_kernels_fp16_fp16.cu",
            "nv_internal/.../moe_gemm_kernels_fp16_uint8.cu",
            "nv_internal/.../moe_gemm_kernels_bf16_bf16.cu",
            "nv_internal/.../moe_gemm_kernels_bf16_uint8.cu",
            
            # FlashInfer PyTorch 绑定
            "fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_sm80_ops.cu",
            "fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu",
            
            # CUTLASS 预生成实例化
            "nv_internal/.../cutlass_kernel_file_4.generated.cu",  # SM80 实例化
            "nv_internal/.../cutlass_kernel_file_5.generated.cu",
            "nv_internal/.../cutlass_kernel_file_6.generated.cu",
            "nv_internal/.../cutlass_kernel_file_7.generated.cu",
            
            # 通用工具库
            "nv_internal/cpp/common/envUtils.cpp",
            "nv_internal/cpp/common/logger.cpp",
            # ... 其他工具文件 ...
        ],
        extra_cuda_cflags=[
            "-DFLASHINFER_ENABLE_TMA_WS=0",  # 禁用 Hopper TMA
            "-DENABLE_BF16",
            "-DENABLE_FP16",
            # Note: FP8 和 FP4 NOT enabled for SM80
            "-DUSING_OSS_CUTLASS_MOE_GEMM",
            "-gencode=arch=compute_80,code=sm_80",  # SM80 代码生成
        ],
        # ... 其他编译标志 ...
    )

@functools.cache
def get_fused_moe_sm80_module():
    """
    缓存 SM80 模块的编译结果
    缓存位置: ~/.cache/flashinfer/fused_moe_sm80/
    """
    module = gen_fused_moe_sm80_module().build_and_load(class_name="FusedMoeRunner")
    
    class MoERunner(TunableRunner):
        """PyTorch 可调整的 MoE 运行器"""
        _runner_dict: Dict[str, object] = dict()
        
        def __init__(self, ...):
            # 初始化 FusedMoeRunner 实例
            pass
    
    return SimpleNamespace(
        cutlass_fused_moe_sm80=cutlass_fused_moe_sm80,  # 注册的自定义操作
    )
```

**关键编译标志解释**:

| 标志 | 说明 |
|------|------|
| `-DFLASHINFER_ENABLE_TMA_WS=0` | SM80 不支持 Hopper TMA，该标志禁用 TMA warp specialized kernels |
| `-DENABLE_BF16` | 启用 BF16 支持 |
| `-DENABLE_FP16` | 启用 FP16 支持 |
| `-DUSING_OSS_CUTLASS_MOE_GEMM` | 使用开源 CUTLASS MOE GEMM |
| `-gencode=arch=compute_80,code=sm_80` | 仅为 SM80 生成优化代码 |

### 2.3 注册的自定义操作

**文件**: `flashinfer/fused_moe.py` (Line 595-735)

```python
@register_custom_op(
    "flashinfer::cutlass_fused_moe_sm80",
    mutates_args=(""),
)
def cutlass_fused_moe_sm80(
    output: torch.Tensor,                      # 输出张量（预分配）
    input: torch.Tensor,                       # 输入激活值 [seq_len, hidden_size]
    token_selected_experts: torch.Tensor,      # [seq_len, top_k]
    token_final_scales: torch.Tensor,          # [seq_len, top_k]
    fc1_expert_weights: torch.Tensor,          # [num_experts, inter_size, hidden_size]
    fc2_expert_weights: torch.Tensor,          # [num_experts, hidden_size, inter_size]
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],          # 必须为空（SM80 不支持量化）
    input_sf: Optional[torch.Tensor] = None,
    tp_size: int = 1,                          # Tensor Parallelism 大小
    tp_rank: int = 0,
    ep_size: int = 1,                          # Expert Parallelism 大小
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    use_fp8_block_scaling: bool = False,
    use_w4a8_group_scaling: bool = False,
    min_latency_mode: bool = False,
    tune_max_num_tokens: int = 8192,
) -> List[torch.Tensor]:
```

**执行步骤**:

1. **参数验证** (Line 602-630):
   ```cpp
   TORCH_CHECK(!min_latency_mode, "Min latency mode not supported on SM80");
   TORCH_CHECK(token_selected_experts.scalar_type() == c10::ScalarType::Int);
   TORCH_CHECK(token_final_scales.scalar_type() == c10::ScalarType::Float);
   TORCH_CHECK(quant_scales.size() == 0, 
               "INT8 not enabled for SM80 backend");
   ```

2. **提取形状信息** (Line 631-636):
   ```python
   num_tokens = input.size(0)            # seq_len
   hidden_size = input.size(1)           # hidden_size
   inter_size = fc1_expert_weights.size(1) / mInnerDimMultiplier
   num_experts = fc1_expert_weights.size(0)
   top_k = token_selected_experts.size(1)
   ```

3. **创建运行器** (Line 637-650):
   ```python
   moe_runner = MoERunner(
       x_dtype=input.dtype,                    # FP16 或 BF16
       weight_dtype=fc1_expert_weights.dtype,
       output_dtype=output_dtype,
       top_k=token_selected_experts.size(1),
       tp_size=tp_size, tp_rank=tp_rank,
       ep_size=ep_size, ep_rank=ep_rank,
       cluster_size=cluster_size, cluster_rank=cluster_rank,
       use_fp8_block_scaling=False,
       use_w4a8_group_scaling=False,
   )
   ```

4. **GEMM Tactic 选择**（自动调整） (Line 651-675):
   ```python
   tuner = AutoTuner.get()
   
   # GEMM1 tactic 选择
   _, gemm_tactic_1 = tuner.choose_one(
       "trtllm::fused_moe::gemm1",
       [moe_runner],
       tuning_config,
       [input, fc1_expert_weights, fc2_expert_weights, min_latency_tensor],
       gemm_idx=1,
   )
   
   # GEMM2 tactic 选择
   _, gemm_tactic_2 = tuner.choose_one(
       "trtllm::fused_moe::gemm2",
       [moe_runner],
       tuning_config,
       [input, fc1_expert_weights, fc2_expert_weights, min_latency_tensor],
       gemm_idx=2,
   )
   ```
   
   **Tactic 说明**: 不同的 CUTLASS GEMM 配置（瓦片大小、stage 数等）有不同的性能特性。AutoTuner 通过实际运行来选择最优的 tactic。

5. **执行 MOE 运算** (Line 676-695):
   ```python
   result = moe_runner._fused_moe_runner.run_moe(
       output,
       input,
       token_selected_experts,
       token_final_scales,
       fc1_expert_weights,
       fc2_expert_weights,
       quant_scales,
       input_sf,
       tp_size, tp_rank, ep_size, ep_rank,
       cluster_size, cluster_rank,
       min_latency_mode,
       [gemm_tactic_1, gemm_tactic_2],  # 应用调整后的 tactics
   )
   ```

### 2.4 C++ FusedMoeRunner 实现

**文件**: `csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_sm80_ops.cu` (Line 97-312)

#### 初始化阶段 (Constructor)

```cpp
class FusedMoeRunner : public torch::CustomClassHolder {
 public:
  FusedMoeRunner(c10::ScalarType activation_dtype,
                 c10::ScalarType weight_dtype,
                 c10::ScalarType output_dtype,
                 bool use_fp8_block_scaling,
                 bool use_w4a8_group_scaling) {
    mActivationDtype = activation_dtype;
    mWeightDtype = weight_dtype;
    mOutputDtype = output_dtype;
    
    // SM80 限制检查
    if (use_fp8_block_scaling)
        C10_THROW_ERROR("FP8 Block Scaling not supported on SM80");
    if (use_w4a8_group_scaling)
        C10_THROW_ERROR("W4A8 Group Scaling not supported on SM80");
    
    // 创建对应数据类型的 CUTLASS MOE GEMM 运行器
    if (activation_dtype == c10::ScalarType::Half &&
        weight_dtype == c10::ScalarType::Half) {
        mKernelRunner = std::make_shared<
            kernels::CutlassMoeFCRunner<half, half>>();
    }
#ifdef ENABLE_BF16
    else if (activation_dtype == c10::ScalarType::BFloat16 &&
             weight_dtype == c10::ScalarType::BFloat16) {
        mKernelRunner = std::make_shared<
            kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
    }
#endif
    else {
        C10_THROW_ERROR("SM80 supports FP16 or BF16 only");
    }
    
    // 获取可用的 GEMM tactics
    mGemmTactics = mKernelRunner->getTactics();
  }
};
```

**支持的数据类型**:
- `<half, half>`: FP16 激活 × FP16 权重
- `<__nv_bfloat16, __nv_bfloat16>`: BF16 激活 × BF16 权重

#### 执行阶段 (run_moe)

```cpp
torch::Tensor run_moe(torch::Tensor& output,
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
                      torch::List<int64_t> gemm_tactics) {
    // 验证输入
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(!min_latency_mode);
    
    // 提取维度信息
    int64_t num_tokens = x.size(0);
    int64_t hidden_size = x.size(1);
    int64_t inter_size = fc1_expert_weights.size(1);
    int64_t num_experts = fc1_expert_weights.size(0);
    int64_t top_k = token_selected_experts.size(1);
    
    // 获取 CUDA 流
    auto stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();
    
    // 构造并行化配置
    kernels::MOEParallelismConfig parallelism_config(
        tp_size, tp_rank, ep_size, ep_rank);
    
    // 激活类型固定为 SwiGLU
    auto activation_type = tensorrt_llm::ActivationType::Swiglu;
    
    // 应用自动调整选择的 tactics
    apply_gemm_tactics(gemm_tactics);
    
    // 计算工作空间大小
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
    
    // MOE 映射表大小
    size_t map_bytes = static_cast<size_t>(top_k * num_tokens * sizeof(int));
    
    // 分配工作空间
    std::array<size_t, 2> workspace_sizes{{moe_workspace_size, map_bytes}};
    size_t total_workspace_size = 
        tensorrt_llm::common::calculateTotalWorkspaceSize(
            workspace_sizes.data(), workspace_sizes.size());
    
    auto options = torch::TensorOptions()
        .dtype(torch::kInt8)
        .device(x.device());
    ensure_workspace(options, total_workspace_size);
    
    // 工作空间指针管理
    auto* base_ptr = static_cast<int8_t*>(mWorkspace.data_ptr());
    auto* moe_ptr = base_ptr;
    auto* map_ptr = tensorrt_llm::common::nextWorkspacePtr(
        base_ptr, moe_workspace_size);
    
    // 创建量化和 LoRA 参数（SM80 不使用）
    kernels::QuantParams quant_params{};
    kernels::LoraParams lora_params{};
    kernels::MoeMinLatencyParams min_latency_params{};
    
    // 执行主要 MOE 计算
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
        reinterpret_cast<char*>(moe_ptr),
        output.data_ptr(),
        reinterpret_cast<int*>(map_ptr),
        parallelism_config,
        false,  // use_lora
        lora_params,
        mUseFp8BlockScaling,
        false,  // min_latency_mode
        min_latency_params,
        stream
    );
    
    return output;
}
```

---

## 第三部分：Device 侧 CUDA Kernel 执行流程

### 3.1 CUTLASS MOE GEMM 架构

SM80 使用基于 CUTLASS 的分组 GEMM (Grouped GEMM) 实现 MOE 计算，这是一种将多个不同形状的 GEMM 操作融合成单个 kernel 的技术。

```
输入布局：
┌────────────────────────────────────────────────────┐
│ Token 0: Expert A        │                        │
│ Token 1: Expert A, B     │ [seq_len, hidden_size] │
│ Token 2: Expert B, C     │                        │
│ Token 3: Expert C        │                        │
└────────────────────────────────────────────────────┘

GEMM1 操作:
Input [seq_len, hidden_size] ×  FC1 Weights [num_experts, inter_size, hidden_size]ᵀ
                                 ↓
                        Intermediate [seq_len, inter_size]
                        + SwiGLU 激活

GEMM2 操作:
Intermediate [seq_len, inter_size] × FC2 Weights [num_experts, hidden_size, inter_size]ᵀ
                                      ↓
                             Output [seq_len, hidden_size]
                             + Router Scale 应用
```

### 3.2 SM80 Kernel Launcher

**文件**: `csrc/nv_internal/tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl`

```cpp
void sm80_generic_fused_moe_gemm_kernelLauncher(
    ElementType_ const* A,                    // 激活值
    CutlassWeightType_ const* B,              // 权重
    ElementType_ const* biases,
    bool bias_is_broadcast,
    ElementType_* C,                          // 输出
    int64_t const* total_tokens_including_expert,  // 每个专家的 token 计数
    int64_t num_rows,
    int64_t gemm_n,                           // 输出列数
    int64_t gemm_k,                           // 共享维度
    int num_experts,
    int multi_processor_count,                // GPU 流多处理器数量
    cudaStream_t stream,
    int* kernel_occupancy
) {
    // 定义 CUTLASS Kernel 类型
    constexpr auto activation_type = fused_moe::EpilogueRouting<EpilogueTag>(true);
    using GemmType = fused_moe::Fused_Moe_Kernel_sm80<
        ElementType_,                // 激活值类型
        CutlassWeightType_,          // 权重类型
        ElementType_,                // 输出类型
        MaxTileM_,                   // 瓦片 M 维度（行）
        TileN_,                      // 瓦片 N 维度（列）
        TileK_,                      // 瓦片 K 维度（共享）
        Stages_,                     // 流水线 stages
        activation_type
    >;
    
    // 计算占用率和最优线程块数量
    int occupancy = std::min(2, fused_moe::fused_gemm_maximum_active_blocks<GemmType>());
    int threadblock_count = multi_processor_count * occupancy;
    
    // 创建 Kernel 参数
    using Arguments = typename GemmType::Arguments;
    Arguments args{
        {
            const_cast<ElementType_*>(A),
            const_cast<CutlassWeightType_*>(B),
            const_cast<ElementType_*>(biases),
            reinterpret_cast<ElementType_*>(C),
            total_tokens_including_expert,
            static_cast<int>(gemm_n),
            static_cast<int>(gemm_k),
            num_experts,
            bias_is_broadcast
        },
        num_experts,
        threadblock_count
    };
    
    auto params = GemmType::to_underlying_arguments(args);
    
    // 如果需要大量共享内存，设置属性
    if (GemmType::kSmemSize >= (48 << 10)) {
        cudaFuncSetAttribute(
            fused_moe::run_global<GemmType>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            GemmType::kSmemSize
        );
    }
    
    // Launch kernel
    dim3 grid(params.threadblock_count, 1, 1);
    dim3 block(GemmType::kThreadCount);
    fused_moe::run_global<GemmType><<<grid, block, GemmType::kSmemSize, stream>>>(params);
}
```

### 3.3 CUTLASS Fused MOE Kernel 核心 (SM80)

**文件**: `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`

```cpp
template <typename ElementType_,
          typename CutlassWeightType_,
          int MaxTileM_,
          int TileN_,
          int TileK_,
          int Stages_,
          typename EpilogueTag>
class Fused_Moe_Kernel_sm80 {
    // CUTLASS 集成 GEM 配置
    // MaxTileM_: 每个线程块处理的行数（通常 64-256）
    // TileN_: 每个线程块处理的列数（通常 256）
    // TileK_: 每个 stage 处理的共享维度（通常 64）
    // Stages_: 流水线深度（通常 2-3）
};
```

**SM80 核心特性**:

| 特性 | 说明 |
|------|------|
| **Tensor Core** | 每个 SM80 有 64 个 FP16 Tensor Cores（相对于 SM90 较少） |
| **Shared Memory** | 96 KB（可配置 48 KB 或 96 KB） |
| **Register File** | 256 KB per SM |
| **组件化 GEMM** | 对标准 CUTLASS GEMM_GROUPED 进行了优化，支持变量大小批处理 |
| **流水线** | 典型 2-3 stage 流水线 |

### 3.4 GEMM1 执行流程（激活 → 中间）

```
Input Shape: [num_tokens, hidden_size]
FC1 Weights: [num_experts, inter_size, hidden_size]

对每个 token：
  1. 根据 token_selected_experts 找到该 token 的专家
  2. 加载对应专家的 FC1 权重
  3. 执行 GEMM: [1, hidden_size] × [hidden_size, inter_size]ᵀ = [1, inter_size]
  4. 应用 SwiGLU 激活: output = silu(x₁) * x₂
     其中 x₁, x₂ 来自 inter_size 的两部分
  5. 存储中间结果到工作空间

线程分配:
  - 每个 thread block 处理多个 tokens 和多个 experts
  - 利用 Tensor Cores 加速 GEMM
  - 共享内存缓存权重和中间值
```

### 3.5 GEMM2 执行流程（中间 → 输出）

```
Intermediate Shape: [num_tokens, inter_size]
FC2 Weights: [num_experts, hidden_size, inter_size]

对每个 token：
  1. 加载该 token 的中间结果（来自 GEMM1）
  2. 根据 token_selected_experts 加载该 token 的专家 FC2 权重
  3. 执行 GEMM: [1, inter_size] × [inter_size, hidden_size]ᵀ = [1, hidden_size]
  4. 应用 Router Scale（来自 token_final_scales）
  5. 累加到最终输出（如果多个专家，则加权求和）

线程分配（类似 GEMM1）:
  - Thread block 级别并行
  - Tensor Core 利用
  - 共享内存优化
```

### 3.6 Kernel 参数结构

```cpp
struct Fused_Moe_Arguments {
    // GEMM 基础参数
    typename GemmKernel::Arguments args;
    
    // MOE 特定参数
    int num_experts;                          // 总专家数
    int threadblock_count;                    // 线程块总数
    
    // 数据指针
    ElementType_* activation;                 // [num_tokens, hidden_size]
    CutlassWeightType_* weights;              // [num_experts, ...]
    int64_t* token_expert_offset;             // [num_experts + 1]
    ElementType_* output;                     // [num_tokens, output_size]
};
```

### 3.7 Shared Memory 使用

SM80 典型共享内存分配（96 KB）:

```
┌─────────────────────────────────────────────┐
│  GEMM Tile Cache (A & B)        ~60 KB      │  用于缓存当前 tile
├─────────────────────────────────────────────┤
│  Epilogue Shared Memory         ~20 KB      │  用于归约和激活
├─────────────────────────────────────────────┤
│  Router Scale Cache              ~8 KB      │  缓存 token_final_scales
├─────────────────────────────────────────────┤
│  其他（Expert Index，etc）        ~8 KB      │
└─────────────────────────────────────────────┘
总计: ~96 KB
```

### 3.8 Occupancy 分析

SM80 每个 SM 特性:
- 最大活跃线程块: 32
- 每个 SM 的最大线程: 2048
- Warp 大小: 32

典型配置:
```
Tile Size: 128x256x64 (M x N x K)
Threads per block: 512 (16 warps)
Shared Memory: 48-96 KB
预期占用率: 2-3 个线程块/SM
```

---

## 第四部分：执行总结与关键优化

### 4.1 SM80 与 SM90+ 的主要差异

| 特性 | SM80 (Ampere) | SM90 (Hopper) | SM100 (Blackwell) |
|------|---------------|---------------|-------------------|
| **Tensor Cores** | 64/SM | 128/SM | 256/SM |
| **GEMM 方式** | CUTLASS Grouped | TMA + Warp Specialized | 增强型 TMA |
| **FP8 支持** | 有限 | 完全 | 完全 |
| **FP4 支持** | ❌ | ✓ | ✓ |
| **Block Scale** | ❌ | ✓ | ✓ |
| **Min Latency Mode** | ❌ | ✓ | ✓ |
| **共享内存/SM** | 96 KB | 144 KB | 等同 SM90 |

### 4.2 SM80 MOE 性能优化要点

1. **Tactic 自动选择**:
   - AutoTuner 在第一次运行时测试不同的 GEMM 配置
   - 结果缓存在内存中，后续调用复用
   - 避免每次运行都进行完整测试

2. **工作空间预分配**:
   ```
   大小 = MOE_WORKSPACE + TOKEN_EXPERT_MAP
        = (中间值缓冲) + (expert routing map)
   ```

3. **流水线优化**:
   - 2 stage 流水线（相对于 SM90 的 4-5 stage）
   - 较低的寄存器压力

4. **共享内存管理**:
   - 48 KB 基础模式（可配置到 96 KB）
   - 优先缓存频繁访问的权重切片

### 4.3 典型性能特性

对于配置：
- Input: [128, 4096] (seq_len=128, hidden_size=4096)
- Experts: 8, top_k=2
- inter_size: 14336
- Data type: FP16

**预期性能** (相对基准):
- GEMM1: ~8-10 TFLOPs (相对于理论 ~200 TFLOP/s 峰值)
- GEMM2: ~8-10 TFLOPs
- 端到端: 150-300 ms (取决于 GPU 型号，如 A10/A100)

### 4.4 限制和注意事项

**SM80 不支持**:
1. ✗ FP8 block scaling（需要 Hopper+ 的动态量化）
2. ✗ W4A8 group scaling（复杂的量化格式）
3. ✗ Min latency mode（依赖 TMA warp specialized）
4. ✗ FP4 数据类型（精度不足，且需要特殊硬件支持）

**推荐用途**:
- FP16/BF16 标准精度计算
- 推理工作负载（不需要梯度）
- 模型并行（Tensor/Expert 并行）

---

## 第五部分：实际执行示例

### 5.1 测试代码执行追踪

从 `test_sm80_compile.py`:

```python
# 1. 导入和 GPU 检测
import flashinfer
major, minor = torch.cuda.get_device_capability(device)
# 输出: GPU: SM80 (A100-80GB, L4, etc.)

# 2. 编译 SM80 模块
from flashinfer.fused_moe import gen_fused_moe_sm80_module
module_spec = gen_fused_moe_sm80_module()
compiled_module = module_spec.build_and_load(class_name="FusedMoeRunner")
# 首次调用时编译 (~2-5 分钟)
# 结果缓存在: ~/.cache/flashinfer/fused_moe_sm80/

# 3. 创建测试张量
input_tensor = torch.randn(4, 128, dtype=torch.float16, device=device)
# Shape: [seq_len=4, hidden_size=128]

fc1_expert_weights = torch.randn(8, 256, 128, dtype=torch.float16, device=device)
# Shape: [num_experts=8, inter_size=256, hidden_size=128]

fc2_expert_weights = torch.randn(8, 128, 256, dtype=torch.float16, device=device)
# Shape: [num_experts=8, hidden_size=128, inter_size=256]

token_selected_experts = torch.randint(0, 8, (4, 2), dtype=torch.int32, device=device)
# Shape: [seq_len=4, top_k=2]

token_final_scales = torch.rand(4, 2, dtype=torch.float32, device=device)
# Shape: [seq_len=4, top_k=2]

# 4. 调用 MOE
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
# 输出: [4, 128] (shape 与预期一致)

print(f"Output shape: {output.shape}")
# Output shape: torch.Size([4, 128])
```

### 5.2 内核执行时间线

```
Timeline on NVIDIA A100-80GB (SM80):

T=0 μs      │ Host: cutlass_fused_moe() 调用
            │ Python → C++ 转换
            │
T~100 μs    │ Host: 参数验证完成
            │ Device: 工作空间分配（H2D 传输）
            │
T~500 μs    │ Device: GEMM1 Kernel Launch
            │ Grid: [occupancy, 1, 1]
            │ Block: [512 threads]
            │ Shared Memory: 48-96 KB
            │
T~500 μs    │ Device: GEMM1 执行
T~5 ms      │ - Tile 装载
            │ - GEMM 计算（通过 Tensor Cores）
            │ - SwiGLU 激活（epilogue）
            │ - 结果写回
            │
T~5 ms      │ Device: __syncthreads() + Barrier
            │ GEMM1 完成，所有线程块同步
            │
T~5 ms      │ Device: GEMM2 Kernel Launch（可能异步）
            │
T~5-10 ms   │ Device: GEMM2 执行（类似 GEMM1）
            │
T~10 ms     │ Device: Router Scale 应用
            │ （通常融合在 GEMM2 epilogue 中）
            │
T~10-15 ms  │ Host: cudaDeviceSynchronize() 
            │ (如果显式同步)
            │
T~15 ms     │ Python: 结果返回给 PyTorch
```

---

## 第六部分：源代码树映射

```
flashinfer/
├── flashinfer/
│   ├── __init__.py
│   │   └── from .fused_moe import cutlass_fused_moe  (L52)
│   │
│   ├── fused_moe.py  ★★★ 主要 Python 接口
│   │   ├── gen_fused_moe_sm100_module()  (L34-109)
│   │   ├── gen_fused_moe_sm80_module()   (L125-209)  ★ SM80 模块生成
│   │   ├── get_fused_moe_sm80_module()   (L340-...)  ★ 缓存的模块加载
│   │   ├── cutlass_fused_moe_sm80()      (L595-735)  ★ 注册的 C++ 操作
│   │   ├── cutlass_fused_moe()           (L741-915)  ★ 主 API（架构选择）
│   │   └── ...
│   │
│   ├── jit/
│   │   ├── __init__.py
│   │   ├── env.py      # JIT 环境（缓存目录）
│   │   └── ...
│   │
│   ├── utils.py
│   │   ├── register_custom_op()      # 操作注册
│   │   ├── register_fake_op()        # 伪操作注册（文档构建）
│   │   ├── get_compute_capability()  # GPU 架构检测
│   │   └── ...
│   │
│   └── autotuner.py     # 自动调优框架
│
├── csrc/
│   ├── fused_moe/
│   │   └── cutlass_backend/
│   │       ├── flashinfer_cutlass_fused_moe_sm80_ops.cu      ★ SM80 C++ 实现
│   │       ├── flashinfer_cutlass_fused_moe_sm100_ops.cu
│   │       ├── cutlass_fused_moe_kernels.cuh                ★ 核心 GEMM kernels
│   │       └── cutlass_fused_moe_instantiation.cu           ★ 模板实例化
│   │
│   └── nv_internal/
│       ├── tensorrt_llm/
│       │   ├── common/
│       │   │   ├── envUtils.cpp
│       │   │   ├── logger.cpp
│       │   │   ├── memoryUtils.cu
│       │   │   └── ...
│       │   │
│       │   ├── kernels/
│       │   │   ├── internal_cutlass_kernels/
│       │   │   │   ├── include/
│       │   │   │   │   ├── moe_kernels.h                   ★ MOE 接口
│       │   │   │   │   ├── moe_gemm_kernels.h
│       │   │   │   │   └── ...
│       │   │   │   │
│       │   │   │   ├── src/
│       │   │   │   │   ├── moe_gemm/
│       │   │   │   │   │   ├── moe_gemm_kernels_fp16_fp16.cu      ★ FP16 GEMM
│       │   │   │   │   │   ├── moe_gemm_kernels_fp16_uint8.cu
│       │   │   │   │   │   ├── moe_gemm_kernels_bf16_bf16.cu      ★ BF16 GEMM
│       │   │   │   │   │   ├── moe_gemm_kernels_bf16_uint8.cu
│       │   │   │   │   │   ├── moe_gemm_tma_warp_specialized_input.cu
│       │   │   │   │   │   │
│       │   │   │   │   │   └── launchers/
│       │   │   │   │   │       ├── fused_moe_gemm_launcher_sm80.h   ★ SM80 Launcher
│       │   │   │   │   │       ├── fused_moe_gemm_launcher_sm80.inl ★ SM80 实现
│       │   │   │   │   │       ├── moe_gemm_tma_ws_launcher.h
│       │   │   │   │   │       └── ...
│       │   │   │   │   │
│       │   │   │   │   └── ...
│       │   │   │   │
│       │   │   │   └── ...
│       │   │   │
│       │   │   ├── preQuantScaleKernel.cu
│       │   │   ├── cutlass_kernels/
│       │   │   │   ├── cutlass_heuristic.cpp
│       │   │   │   └── ...
│       │   │   │
│       │   │   └── ...
│       │   │
│       │   └── ...
│       │
│       ├── cpp/
│       │   └── common/
│       │       ├── envUtils.cpp
│       │       ├── logger.cpp
│       │       ├── tllmException.cpp
│       │       └── ...
│       │
│       └── ...
│
├── tests/
│   ├── test_trtllm_cutlass_fused_moe.py    # 功能测试
│   └── ...
│
└── test_sm80_compile.py                     ★ 你的测试文件
```

---

## 第七部分：关键数据结构

### 7.1 MOEParallelismConfig

```cpp
struct MOEParallelismConfig {
  int tp_size = 1;        // Tensor Parallelism 大小
  int tp_rank = 0;        // 本地 rank (0 to tp_size-1)
  int ep_size = 1;        // Expert Parallelism 大小
  int ep_rank = 0;        // 本地 expert rank
  int cluster_size = 1;   // 集群大小（用于高级分布式）
  int cluster_rank = 0;   // 本地集群 rank
};
```

### 7.2 QuantParams

```cpp
struct QuantParams {
  // SM80 通常使用空配置
  // (所有量化指针为 nullptr)
  struct {
    void const* fc1_weight_scales = nullptr;
    void const* fc2_weight_scales = nullptr;
  } wo;  // 权重量化参数
  
  struct {
    float const* dequant_fc1 = nullptr;
    float const* quant_fc2 = nullptr;
    // ... 其他 FP8 参数
  } fp8;  // FP8 参数（SM80 不使用）
  
  // ... 其他量化相关参数
};
```

### 7.3 CutlassGemmConfig (Tactic)

```cpp
struct CutlassGemmConfig {
  int tile_m;           // Block tile M 维度
  int tile_n;           // Block tile N 维度
  int tile_k;           // Block tile K 维度
  int stages;           // Pipeline stages
  // ... 其他配置参数
};
```

---

## 第八部分：常见问题与故障排查

### Q1: 为什么 SM80 不支持 FP8 block scaling?

**A**: SM80 (Ampere) 的 FP8 支持方式不同于 Hopper (SM90+)：
- SM80 需要手动的量化/反量化操作
- FP8 block scaling 需要动态的块级缩放索引，这在 SM80 kernel 中难以高效实现
- Hopper 引入了 TMA (Tensor Memory Accelerator) 可以更高效地处理这类操作

### Q2: 什么是 "Tactic"?

**A**: Tactic 是 CUTLASS GEMM 的一种配置，包括：
- 线程块瓦片大小 (M, N, K)
- 流水线深度 (stages)
- 其他硬件配置

不同的 tactic 有不同的性能特性。AutoTuner 通过实际测试找出最优的。

### Q3: 为什么第一次运行很慢？

**A**: 三个原因：
1. **JIT 编译**: 首次调用时编译 CUDA 代码 (~2-5 分钟)
2. **Auto-tuning**: AutoTuner 测试不同的 tactics (~1-2 分钟)
3. **GPU 预热**: CUDA 上下文初始化

后续调用从缓存加载，速度快 100x。

### Q4: 如何确认代码在 SM80 上运行？

**A**:
```python
import torch
print(torch.cuda.get_device_capability())  # 应该返回 (8, x)
print(torch.cuda.get_device_name())        # 应该是 A10, A30, A100 等
```

### Q5: GEMM1 和 GEMM2 有什么区别？

**A**:
```
GEMM1: [seq_len, hidden] × [inter_size, hidden]ᵀ → [seq_len, inter_size]
       - 应用激活函数 (SwiGLU)
       - 扩展维度

GEMM2: [seq_len, inter_size] × [hidden, inter_size]ᵀ → [seq_len, hidden]
       - 应用 router scaling
       - 返回到原始维度
       - 如果多个专家则求加权和
```

---

## 结论

SM80 上的 `cutlass_fused_moe` 实现：

1. **架构选择明智**: 在 SM80 上禁用了 FP8/FP4 等需要 Hopper 硬件支持的特性
2. **性能考虑周全**: 使用 CUTLASS Grouped GEMM，充分利用 Tensor Cores
3. **自动优化**: AutoTuner 自动选择最优的 kernel 配置
4. **兼容性好**: 支持 FP16/BF16，适配现有模型
5. **扩展性强**: 支持 Tensor/Expert 并行

对于 A100、A10 等 SM80 设备，这个实现提供了高效、开箱即用的 MOE 推理能力。
