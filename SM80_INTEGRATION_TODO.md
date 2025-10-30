# SM80 简化 MoE 集成 - 待办事项

## 状态说明
- ✅ 已完成
- ⏳ 进行中  
- ❌ 待开始
- 🔒 被阻塞

---

## 阶段 1: 核心集成 (P0 - 必须完成)

### 1.1 C++ 扩展实现

- [✅] 创建 `flashinfer_simple_moe_sm80_ops.cu`
  - [✅] 定义 `FusedMoeRunner` 类
  - [✅] 实现 `get_tactic_num()` (返回 1)
  - [✅] 实现 `run_gemm_profile()` (空操作)
  - [✅] 实现 `run_moe()` 主函数
  - [✅] 添加 FP16/BF16/INT8 类型分发
  - [✅] 注册到 `torch.classes.fused_moe_sm80`

### 1.2 Python API 修改

- [✅] 修改 `flashinfer/fused_moe.py`
  - [✅] 更新 `gen_fused_moe_sm80_module()` 
  - [✅] 移除 TRT-LLM/CUTLASS 依赖
  - [✅] 添加简化实现路径
  - [✅] 保持现有 `cutlass_fused_moe()` API 不变

### 1.3 GEMM 实现 (关键路径)

- [❌] **实现 cuBLAS GEMM 后端**
  - [❌] 创建 `csrc/fused_moe_simple/cublas_gemm_impl.cu`
  - [❌] 实现 `simple_moe_gemm1()` 使用 `cublasGemmEx`
  - [❌] 实现 `simple_moe_gemm2()` 使用 `cublasGemmEx`
  - [❌] 处理分组 GEMM (每个专家独立调用)
  - [❌] 添加错误处理

**优先级**: 🔴 **最高** - 阻塞所有测试

**当前阻塞**: 无法运行任何实际测试，因为没有 GEMM 实现

**建议实现**:
```cpp
// csrc/fused_moe_simple/cublas_gemm_impl.cu
#include <cublas_v2.h>
#include <cuda_fp16.h>

extern "C" void simple_moe_gemm1(
    void* output, void const* input, void const* weights, void const* bias,
    int64_t const* expert_offsets, int64_t M, int64_t N, int64_t K,
    int num_experts, bool is_fp16, cudaStream_t stream) {
  
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);
  
  // 对每个专家执行 GEMM
  for (int e = 0; e < num_experts; ++e) {
    int64_t start = expert_offsets[e];
    int64_t end = expert_offsets[e + 1];
    int64_t m = end - start;
    
    if (m == 0) continue;
    
    // GEMM: C = A @ B^T
    // A: [m, K] - 输入
    // B: [N, K] - 权重 (需要转置)
    // C: [m, N] - 输出
    
    if (is_fp16) {
      half* C = static_cast<half*>(output) + start * N;
      half const* A = static_cast<half const*>(input) + start * K;
      half const* B = static_cast<half const*>(weights) + e * N * K;
      
      float alpha = 1.0f, beta = 0.0f;
      cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          N, m, K,
          &alpha,
          B, CUDA_R_16F, K,
          A, CUDA_R_16F, K,
          &beta,
          C, CUDA_R_16F, N,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP
      );
    }
  }
  
  cublasDestroy(handle);
}

// 类似实现 simple_moe_gemm2()
```

### 1.4 编译系统

- [⏳] 验证 JIT 编译
  - [❌] 测试编译时间 (目标 < 1 分钟)
  - [❌] 测试缓存机制
  - [❌] 确保在不同 CUDA 版本下编译

**预期问题**:
1. Thrust lambda 可能在某些 CUDA 版本编译失败
2. 需要链接 cuBLAS 库 (`-lcublas`)

---

##阶段 2: 测试与验证 (P0)

### 2.1 单元测试

- [❌] **创建基础测试**
  - [❌] 测试 FP16 forward pass
  - [❌] 测试 BF16 forward pass
  - [❌] 测试 INT8 forward pass (量化)
  - [❌] 测试不同 batch size (1, 16, 128, 1024)
  - [❌] 测试不同专家数量 (4, 8, 16, 64)
  - [❌] 测试不同 top-k (1, 2, 4, 8)

**测试脚本模板**:
```python
# tests/test_sm80_simple_moe.py
import torch
import pytest
import flashinfer

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [16, 128, 1024])
@pytest.mark.parametrize("num_experts", [4, 8, 16])
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_sm80_moe_basic(dtype, num_tokens, num_experts, top_k):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda:0")
    hidden_size = 512
    intermediate_size = 2048
    
    # 创建输入
    input_tensor = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    token_selected_experts = torch.randint(0, num_experts, (num_tokens, top_k), 
                                           dtype=torch.int32, device=device)
    token_final_scales = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
    
    # 创建权重
    fc1_weights = torch.randn(num_experts, 2 * intermediate_size, hidden_size,
                              dtype=dtype, device=device)
    fc2_weights = torch.randn(num_experts, hidden_size, intermediate_size,
                              dtype=dtype, device=device)
    
    # 执行
    output = flashinfer.cutlass_fused_moe(
        input=input_tensor,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        fc1_expert_weights=fc1_weights,
        fc2_expert_weights=fc2_weights,
        output_dtype=dtype,
        quant_scales=[],
    )
    
    # 验证
    assert output.shape == (num_tokens, hidden_size)
    assert output.dtype == dtype
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
```

### 2.2 数值正确性验证

- [❌] **与参考实现对比**
  - [❌] 实现 PyTorch 原生 MoE 参考实现
  - [❌] 对比输出差异 (相对误差 < 1e-3)
  - [❌] 测试边界情况 (0 tokens, 单个 expert, 等等)

**参考实现**:
```python
def pytorch_reference_moe(input, token_selected_experts, token_final_scales,
                         fc1_weights, fc2_weights):
    """纯 PyTorch 实现用于验证正确性"""
    num_tokens, hidden_size = input.shape
    num_experts, inter_size, _ = fc1_weights.shape
    top_k = token_selected_experts.shape[1]
    
    output = torch.zeros(num_tokens, hidden_size, dtype=input.dtype, device=input.device)
    
    for token_idx in range(num_tokens):
        for k in range(top_k):
            expert_id = token_selected_experts[token_idx, k].item()
            weight = token_final_scales[token_idx, k].item()
            
            # GEMM1 + Activation
            hidden = torch.matmul(input[token_idx:token_idx+1], fc1_weights[expert_id].t())
            gate, up = hidden.split(inter_size // 2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            
            # GEMM2
            expert_out = torch.matmul(activated, fc2_weights[expert_id].t())
            
            # 累加
            output[token_idx] += weight * expert_out.squeeze(0)
    
    return output
```

### 2.3 错误处理测试

- [❌] 测试无效输入
  - [❌] 空输入张量
  - [❌] 不匹配的形状
  - [❌] 错误的数据类型
  - [❌] 无效的专家索引
  - [❌] top_k > num_experts

---

## 阶段 3: 性能优化 (P1)

### 3.1 GEMM 优化

- [❌] **实现自定义 Tensor Core GEMM**
  - [❌] 基础 Tensor Core GEMM (FP16)
  - [❌] 优化共享内存使用
  - [❌] 实现流水线 (软件流水线)
  - [❌] 添加 Async Copy (SM80 特性)
  - [❌] 调优块大小和线程布局

**性能目标**:
- cuBLAS: 达到 70-90% 的 TRT-LLM 性能
- 自定义内核: 达到 90-100% 的 TRT-LLM 性能

### 3.2 内存优化

- [❌] 工作空间重用
  - [❌] 实现工作空间池
  - [❌] 避免重复分配
  - [❌] 优化峰值内存使用

### 3.3 Kernel Fusion

- [❌] 融合激活函数
  - [❌] Swiglu 融合到 GEMM1
  - [❌] 减少中间缓冲区

### 3.4 性能基准测试

- [❌] **创建基准测试脚本**
  - [❌] 测试不同配置 (batch, experts, hidden_size)
  - [❌] 对比 PyTorch 原生实现
  - [❌] 对比 TRT-LLM (如果可用)
  - [❌] 生成性能报告

**基准脚本模板**:
```python
# benchmarks/bench_sm80_simple_moe.py
import torch
import time
import flashinfer

def benchmark_config(num_tokens, hidden_size, intermediate_size, 
                     num_experts, top_k, dtype, num_warmup=10, num_iters=100):
    device = torch.device("cuda:0")
    
    # 创建输入
    input_tensor = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    # ... (其他输入)
    
    # 预热
    for _ in range(num_warmup):
        output = flashinfer.cutlass_fused_moe(...)
    
    torch.cuda.synchronize()
    
    # 计时
    start = time.perf_counter()
    for _ in range(num_iters):
        output = flashinfer.cutlass_fused_moe(...)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / num_iters
    return avg_time

# 运行多种配置并生成报告
```

---

## 阶段 4: INT8 量化支持 (P1)

### 4.1 INT8 GEMM 实现

- [❌] 实现 INT8 Tensor Core GEMM
  - [❌] GEMM1: INT8 输入 × INT8 权重
  - [❌] GEMM2: INT8 输入 × INT8 权重
  - [❌] 添加反量化步骤

### 4.2 量化工具

- [❌] 权重量化工具
  - [❌] Per-tensor 量化
  - [❌] Per-channel 量化
  - [❌] 校准数据集支持

### 4.3 INT8 测试

- [❌] 创建 INT8 测试套件
- [❌] 验证精度损失 (< 1% 相对误差)

---

## 阶段 5: 文档与示例 (P1)

### 5.1 文档完善

- [✅] 集成方案文档 (`SM80_SIMPLIFIED_MOE_INTEGRATION.md`)
- [⏳] API 文档
  - [❌] 添加到 `docs/` 目录
  - [❌] 生成 API 参考
- [❌] 性能调优指南
- [❌] 故障排查指南

### 5.2 示例代码

- [✅] 基础示例 (`sm80_moe_with_cutlass_fused_moe_api.py`)
- [❌] 端到端示例
  - [❌] 完整的 Transformer 层
  - [❌] 与 Hugging Face 集成
- [❌] 性能对比示例

---

## 阶段 6: 高级特性 (P2 - 可选)

### 6.1 张量并行支持

- [❌] 实现 TP (Tensor Parallelism)
  - [❌] NCCL 集成
  - [❌] All-Reduce 操作
  - [❌] 跨 GPU 测试

### 6.2 专家并行支持

- [❌] 实现 EP (Expert Parallelism)
  - [❌] 专家分片
  - [❌] All-to-All 通信
  - [❌] 负载均衡

### 6.3 更多激活函数

- [❌] 支持更多激活
  - [❌] Gelu
  - [❌] Relu
  - [❌] Geglu

### 6.4 Layer Fusion

- [❌] 融合 RMSNorm
- [❌] 融合 LayerNorm
- [❌] 融合 Dropout

---

## 已知问题跟踪

### 🐛 Issue #1: Thrust Lambda 编译问题
**状态**: 🔒 待解决  
**优先级**: P1  
**描述**: `prepareRouting()` 中的 Thrust lambda 在某些 CUDA 版本无法编译  
**解决方案**: 用显式 CUDA 内核替换 Thrust 排序  
**文件**: `csrc/fused_moe_simple/moe_runner.cuh:prepareRouting()`

### 🐛 Issue #2: INT8 量化精度
**状态**: ❌ 未开始  
**优先级**: P1  
**描述**: INT8 量化路径未实现完整的缩放和零点支持  
**解决方案**: 实现 Per-channel INT8 GEMM  
**文件**: `csrc/fused_moe_simple/flashinfer_simple_moe_sm80_ops.cu`

### 🐛 Issue #3: cuBLAS Handle 重用
**状态**: ❌ 未开始  
**优先级**: P2  
**描述**: 每次 GEMM 调用都创建/销毁 cuBLAS handle，性能开销大  
**解决方案**: 使用线程局部 handle 池  

---

## 里程碑

### M1: 核心功能可用 (目标: 1 周)
- [✅] C++ 扩展实现
- [✅] Python API 集成
- [❌] cuBLAS GEMM 实现 ⬅️ **当前阻塞**
- [❌] 基础测试通过

### M2: 生产就绪 (目标: 2 周)
- [❌] 所有测试通过
- [❌] 性能达到 cuBLAS 水平 (70-90%)
- [❌] 文档完善

### M3: 性能优化 (目标: 4 周)
- [❌] 自定义 GEMM 内核
- [❌] 性能达到 TRT-LLM 水平 (90-100%)
- [❌] INT8 量化支持

---

## 优先级总结

### 🔴 P0 (必须 - 阻塞发布)
1. ❌ 实现 cuBLAS GEMM 后端
2. ❌ 基础单元测试通过
3. ❌ 数值正确性验证

### 🟡 P1 (重要 - 影响可用性)
4. ❌ 性能基准测试
5. ❌ INT8 量化支持
6. ❌ 自定义 GEMM 内核
7. ⏳ API 文档

### 🟢 P2 (可选 - 锦上添花)
8. ❌ 张量并行/专家并行
9. ❌ 更多激活函数
10. ❌ Layer Fusion

---

## 贡献指南

欢迎贡献！优先处理 P0 和 P1 任务。

**如何贡献**:
1. 从待办事项选择一个任务
2. 创建分支: `git checkout -b feature/task-name`
3. 实现并测试
4. 提交 PR 并链接相关 Issue

**联系方式**:
- GitHub Issues: [flashinfer/issues](https://github.com/flashinfer-ai/flashinfer/issues)
- Discord: FlashInfer Community

---

最后更新: 2025-01-XX  
维护者: FlashInfer Team
