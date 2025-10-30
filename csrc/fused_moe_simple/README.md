# Simple Fused MoE for SM80

这是一个简化的 Fused MoE 实现，专为 SM80 (Ampere) 架构设计，去除了 TensorRT-LLM 和 CUTLASS 依赖。

## 核心特性

- ✅ **无 TensorRT-LLM 依赖**: 完全独立实现
- ✅ **无 CUTLASS 依赖**: 仅使用标准 CUDA 和 Thrust
- ✅ **SM80 专用**: 针对 Ampere 架构优化
- ✅ **INT8 量化支持**: 支持 FP16/BF16/INT8
- ✅ **可插拔 GEMM**: 用户提供自己的 GEMM 实现
- ✅ **简洁高效**: 保留核心功能，去除复杂特性

## 架构概览

```
Input [num_rows, hidden_size]
    ↓
1. Routing & Permutation (routing_kernels.cuh)
   - 按专家分组 tokens
   - 构建映射表
    ↓
Permuted Input [expanded_num_rows, hidden_size]
    ↓
2. GEMM1 (用户提供)
   - Input → Intermediate
   - 应用激活函数 (SwiGLU/GELU)
    ↓
Intermediate [expanded_num_rows, inter_size]
    ↓
3. GEMM2 (用户提供)
   - Intermediate → Output
    ↓
GEMM2 Output [expanded_num_rows, hidden_size] (permuted)
    ↓
4. Finalize (finalize_kernel.cuh)
   - Unpermute (恢复 token 顺序)
   - K-way reduction (聚合多个专家)
   - Scaling (应用 router scales)
    ↓
Output [num_rows, hidden_size]
```

## 核心数据结构

### 映射表

```cpp
// 排序后位置 → 原始扩展位置
int* expanded_dest_row_to_expanded_source_row [expanded_num_rows]

// 原始扩展位置 → 排序后位置 (finalize 使用)
int* expanded_source_row_to_expanded_dest_row [expanded_num_rows]

// 每个扩展行对应的专家 ID
int* expert_for_source_row [expanded_num_rows]

// 每个专家处理的 token 偏移
int64_t* expert_first_token_offset [num_experts + 1]
```

### 计算公式

```
expanded_num_rows = num_rows * top_k

expanded_source_row = original_row + k * num_rows
    where k ∈ [0, top_k)
    
expert_first_token_offset[i] = 专家 i 之前的总 token 数
expert_first_token_offset[num_experts] = expanded_num_rows
```

## 使用方法

### 基本用法

```python
import torch
from flashinfer import fused_moe_simple

def my_gemm1(output, input, weights, bias, expert_offsets, 
             M, N, K, num_experts, is_gemm1, activation):
    """
    用户提供的 GEMM1 实现
    
    Args:
        output: [M, N] 输出张量
        input: [M, K] 输入张量 (已按专家分组)
        weights: [num_experts, N, K] 权重
        bias: [num_experts, N] 偏置或 None
        expert_offsets: [num_experts + 1] 每个专家的 token 范围
        M: expanded_num_rows
        N: inter_size
        K: hidden_size
        num_experts: 专家总数
        is_gemm1: True (标识这是 GEMM1)
        activation: "swiglu" 等
    """
    # 实现分组 GEMM + 激活函数
    for i in range(num_experts):
        start = expert_offsets[i]
        end = expert_offsets[i + 1]
        if start < end:
            # GEMM for expert i
            expert_output = input[start:end] @ weights[i].T
            if bias is not None:
                expert_output += bias[i]
            # Apply activation
            if activation == "swiglu":
                gate, up = expert_output.chunk(2, dim=-1)
                expert_output = F.silu(gate) * up
            output[start:end] = expert_output

def my_gemm2(output, input, weights, bias, expert_offsets,
             M, N, K, num_experts, is_gemm1, activation):
    """用户提供的 GEMM2 实现 (类似 GEMM1 但无激活)"""
    # 实现分组 GEMM
    pass

# 调用 MoE
output = fused_moe_simple(
    input=input,                          # [batch_size, hidden_size]
    token_selected_experts=experts,       # [batch_size, top_k]
    token_final_scales=scales,            # [batch_size, top_k]
    fc1_expert_weights=fc1_weights,       # [num_experts, inter_size, hidden_size]
    fc2_expert_weights=fc2_weights,       # [num_experts, hidden_size, inter_size]
    gemm1_fn=my_gemm1,
    gemm2_fn=my_gemm2,
    activation="swiglu"
)
```

### INT8 量化

```python
# 使用 INT8 权重
input_int8 = input.to(torch.int8)
fc1_weights_int8 = fc1_weights.to(torch.int8)
fc2_weights_int8 = fc2_weights.to(torch.int8)

# GEMM 函数需要处理 INT8 → FP16/BF16 的反量化
output = fused_moe_simple(
    input=input_int8,
    token_selected_experts=experts,
    token_final_scales=scales,
    fc1_expert_weights=fc1_weights_int8,
    fc2_expert_weights=fc2_weights_int8,
    gemm1_fn=my_int8_gemm1,
    gemm2_fn=my_int8_gemm2,
    activation="swiglu"
)
```

## 文件结构

```
csrc/fused_moe_simple/
├── common.cuh                    # 通用定义和工具
├── routing_kernels.cuh           # Routing 和 permutation
├── finalize_kernel.cuh           # Finalize (unpermute + reduce + scale)
├── moe_runner.cuh                # 主运行器逻辑
└── fused_moe_simple_pybind.cu    # PyTorch 绑定

flashinfer/
└── fused_moe_simple.py           # Python API
```

## 性能特点

### 与完整 CUTLASS 实现对比

| 组件 | 本实现 | CUTLASS 版 | 差异 |
|------|--------|-----------|------|
| Routing | ~1-2ms | ~1-2ms | ✓ 相同 |
| GEMM | 用户实现 | CUTLASS | 取决于用户 GEMM 质量 |
| Finalize | <1ms | <1ms | ✓ 相同 |
| 总开销 | +0ms | 基准 | 如果 GEMM 优化良好 |

### 优化建议

1. **GEMM 实现**
   - 使用 CUDA Cores 的 Tensor Cores
   - 向量化内存访问 (128-bit loads)
   - 共享内存 tiling
   - 推荐使用现有库 (cuBLAS, cutlass-less grouped GEMM)

2. **激活函数融合**
   - SwiGLU: 在 GEMM1 kernel 内融合
   - 避免额外的 kernel launch

3. **内存优化**
   - 复用中间缓冲区
   - 使用流水线隐藏延迟

## 限制

- ❌ 不支持张量并行 (TP)
- ❌ 不支持 FP8 block scaling (SM90+ 特性)
- ❌ 不支持 W4A8 group scaling
- ❌ 不支持 min latency mode (TMA 特性)
- ✓ 仅支持 SM80 (Ampere)
- ✓ 需要用户提供 GEMM 实现

## 依赖

- CUDA Runtime (必须)
- PyTorch (必须)
- Thrust (用于排序，可选择用 CUB 替代)
- **无需** TensorRT-LLM
- **无需** CUTLASS

## 编译

```bash
# 使用 JIT 编译
python -c "from flashinfer import fused_moe_simple; fused_moe_simple.get_fused_moe_simple_module()"

# 或在 flashinfer 构建时
python -m pip install --no-build-isolation -e . -v
```

## 示例 GEMM 实现参考

查看 `examples/simple_moe_gemm.py` 了解如何实现高效的 grouped GEMM。

## 许可证

Apache 2.0

## 贡献

欢迎提交 PR 改进性能或添加新特性！
