# SM80 (Ampere) Support for cutlass_fused_moe API - Implementation Summary

## 概述

本次实现为 FlashInfer 的 `cutlass_fused_moe` API 添加了对 **SM80 (Ampere)** 架构的支持，包括 NVIDIA A100、A10、A30 等 GPU。

## 实现的文件

### 1. Python 层 (`flashinfer/fused_moe.py`)

#### 新增函数：

- **`gen_fused_moe_sm80_module()`**: 生成 SM80 的 JIT 编译规范
  - 包含 FP16/BF16/FP32 kernel 源文件
  - 不包含 FP8/FP4 相关文件（SM80 不支持）
  - 使用 `-gencode=arch=compute_80,code=sm_80` 等 SM80 系列标志

- **`get_fused_moe_sm80_module()`**: 缓存的模块加载函数
  - 创建 `MoERunner` 类来管理 kernel 运行
  - 包含 SM80 限制检查（不支持 FP8/FP4/min_latency_mode）
  - 实现 `cutlass_fused_moe_sm80` 自定义操作

#### 修改的函数：

- **`cutlass_fused_moe()`**: 添加架构自动检测
  ```python
  major, minor = get_compute_capability(input.device)
  
  if major == 8:  # SM80-SM89
      # 使用 SM80 backend
      return get_fused_moe_sm80_module().cutlass_fused_moe_sm80(...)
  elif major >= 9:  # SM90+
      # 使用 SM100 backend (Hopper/Blackwell)
      return get_fused_moe_sm100_module().cutlass_fused_moe_sm100(...)
  ```

### 2. C++ 绑定层 (`csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_sm80_ops.cu`)

新创建的文件实现：

- **`FusedMoeRunner` 类**: 
  - 支持 FP16/FP16, BF16/BF16
  - 支持 INT8 (UINT8) 量化权重（需要 FP16/BF16 激活）
  - 支持 fused gated activation (Swiglu/Geglu)
  
- **限制检查**:
  - 拒绝 FP8 激活/权重
  - 拒绝 FP4/INT4 权重（编译错误）
  - 拒绝 FP8 block scaling
  - 拒绝 W4A8 group scaling
  - 拒绝 min_latency_mode

- **Torch 库注册**: `TORCH_LIBRARY(fused_moe_sm80, m)`

### 3. 测试文件 (`tests/test_sm80_fused_moe.py`)

包含以下测试用例：

1. **`test_sm80_detection()`**: 检测 SM80 架构
2. **`test_sm80_fused_moe_fp16()`**: FP16 功能测试
3. **`test_sm80_fused_moe_bf16()`**: BF16 功能测试
4. **`test_sm80_unsupported_fp8()`**: 验证正确拒绝 FP8
5. **`test_sm80_unsupported_min_latency_mode()`**: 验证正确拒绝 min_latency_mode

## 支持的功能矩阵

### ✅ SM80 支持的功能

| 功能 | 支持状态 |
|------|---------|
| FP16 激活和权重 | ✅ |
| BF16 激活和权重 | ✅ |
| INT8 (UINT8) 量化权重 (with FP16/BF16 activations) | ✅ |
| Fused gated activation (Swiglu/Geglu) | ✅ |
| 张量并行 (TP) | ✅ |
| 专家并行 (EP) | ✅ |
| Top-K expert 选择 | ✅ |

### ❌ SM80 不支持的功能

| 功能 | 支持状态 | 原因 |
|------|---------|------|
| FP8 激活/权重 | ❌ | 需要 SM89+ (Ada) 或 SM90+ (Hopper) |
| FP4/INT4 权重 | ❌ | 需要 SM90+ (Hopper)，编译时缺少类型定义 |
| FP8 block scaling | ❌ | 需要 SM90+ (Hopper) |
| W4A8 group scaling | ❌ | 需要 SM90+ (Hopper) |
| Min latency mode | ❌ | 需要 SM90+ (Hopper) |
| FP32 激活和权重 | ❌ | 已移除以简化 SM80 支持 |

## 架构对比

```
┌─────────────────────────────────────────────────────────┐
│                   cutlass_fused_moe()                   │
│              (自动检测 GPU 架构并路由)                   │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
┌─────────▼─────────┐          ┌─────────▼──────────┐
│  SM80 Backend     │          │  SM90+ Backend     │
│  (Ampere)         │          │  (Hopper/Blackwell)│
├───────────────────┤          ├────────────────────┤
│ ✅ FP16/BF16      │          │ ✅ FP16/BF16/FP32  │
│ ✅ INT8 (UINT8)   │          │ ✅ FP8/FP4/INT4    │
│ ❌ FP8/FP4/INT4   │          │ ✅ Block scaling   │
│ ❌ Min latency    │          │ ✅ Min latency     │
└───────────────────┘          └────────────────────┘
```

## 使用示例

```python
import torch
from flashinfer import cutlass_fused_moe

# 在 SM80 GPU 上会自动使用 SM80 backend
device = torch.device("cuda:0")

# 输入
input = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device)
token_selected_experts = torch.randint(0, num_experts, (seq_len, top_k), device=device)
token_final_scales = torch.rand(seq_len, top_k, device=device)

# 专家权重
fc1_weights = torch.randn(num_experts, inter_size, hidden_size, dtype=torch.float16, device=device)
fc2_weights = torch.randn(num_experts, hidden_size, inter_size, dtype=torch.float16, device=device)

# 调用 - 自动选择正确的 backend
output = cutlass_fused_moe(
    input=input,
    token_selected_experts=token_selected_experts,
    token_final_scales=token_final_scales,
    fc1_expert_weights=fc1_weights,
    fc2_expert_weights=fc2_weights,
    output_dtype=torch.float16,
    quant_scales=[],
)
```

## 编译和测试

### 编译

```bash
# 设置目标架构
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7"

# 安装（可编辑模式）
python -m pip install --no-build-isolation -e . -v
```

### 测试

```bash
# 运行 SM80 测试
pytest tests/test_sm80_fused_moe.py -v

# 或直接运行
python tests/test_sm80_fused_moe.py
```

## 兼容性

- **CUDA**: 11.8+
- **PyTorch**: 2.0+
- **GPU**: NVIDIA Ampere 架构
  - A100 (SM80)
  - A10 (SM86)
  - A30 (SM80)
  - RTX 30 系列 (SM86)
  - RTX 40 系列也可以使用 SM80 backend，但推荐使用 SM89 优化

## 性能考虑

- SM80 backend 使用 CUTLASS 模板库实现高效 GEMM
- 支持 fused gated activation 减少内存访问
- 对于 FP8/FP4/INT4 workload，建议升级到 Hopper+ GPU
- INT8 (UINT8) 量化权重可在 SM80 上获得良好性能

## 后续工作

1. ✅ SM80 基础支持 (当前实现)
2. 🔄 性能调优和 benchmark
3. 📝 添加更多测试用例
4. 📊 与 SM90+ 性能对比文档

## 贡献者

- 实现日期: 2025-10-17
- 基于 FlashInfer 架构和现有 SM100 实现
