# GEMM2 到 finalizeMoeRoutingKernel 的数据转换详解

## 完整的 Kernel 链

```
┌──────────────────────────────────────────────────────────────┐
│ 工作空间准备                                                 │
│ expanded_num_rows = num_rows * experts_per_token            │
│ [Token Index Mapping] [Expert Index Mapping]                │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼ (GEMM1 已完成，中间结果在缓冲区中)
┌──────────────────────────────────────────────────────────────┐
│ GEMM2 Input Processing                                       │
│ - 加载 Intermediate [expanded_num_rows, inter_size]         │
│ - 按 expanded_num_rows 重新索引（排列）                     │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│ GEMM2 Kernel Execution                                       │
│                                                              │
│ 对每个 thread block (token 块):                             │
│  1. 加载输入 tile [MaxM, inter_size]                       │
│  2. 加载权重 tile [hidden_size, inter_size]ᵀ             │
│  3. 执行 GEMM: output = input @ weights ᵀ                 │
│  4. 应用 bias + identity 激活                             │
│  5. 输出 [expanded_num_rows, hidden_size] (Permuted)      │
│                                                              │
│ ⚠️ 关键特性:                                               │
│  - 输出顺序是 PERMUTED（按专家组织）                      │
│  - 每个专家的 token 被分组在一起                           │
│  - 原始 token 顺序丢失                                    │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼ cudaDeviceSynchronize() [等待 GEMM2 完成]
┌──────────────────────────────────────────────────────────────┐
│ Permuted Output Layout (GEMM2 结果)                         │
│                                                              │
│ 原始数据:                                                   │
│  Token 0 (Expert A, B) → 中间值                            │
│  Token 1 (Expert A, C) → 中间值                            │
│  Token 2 (Expert B, C) → 中间值                            │
│                                                              │
│ GEMM2 后 (Permuted):                                       │
│  [Expert A 的所有 tokens] ← Token 0, Token 1              │
│  [Expert B 的所有 tokens] ← Token 0, Token 2              │
│  [Expert C 的所有 tokens] ← Token 1, Token 2              │
│                                                              │
│ 内存布局:                                                   │
│ gemm_output = [                                            │
│   Token0@ExpertA [h0, h1, h2, ...],   // hidden_size       │
│   Token1@ExpertA [h0, h1, h2, ...],   // hidden_size       │
│   Token0@ExpertB [h0, h1, h2, ...],   // hidden_size       │
│   Token2@ExpertB [h0, h1, h2, ...],   // hidden_size       │
│   ... (共 expanded_num_rows 行)                           │
│ ]                                                           │
│                                                              │
│ expanded_source_row_to_expanded_dest_row:                 │
│ [0, 1, 2, 3, 4, 5, ...]  ← 原始排列顺序索引             │
│                                                              │
│ expert_for_source_row:                                    │
│ [0, 0, 1, 1, 2, 2, ...]  ← 对应的专家 ID                │
│                                                              │
│ token_final_scales:                                       │
│ [s0_a, s0_b, s1_a, s1_c, s2_b, s2_c, ...]  ← router scales│
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼ finalizeMoeRoutingKernelLauncher 调用
┌──────────────────────────────────────────────────────────────┐
│ finalizeMoeRoutingKernel 执行                               │
│                                                              │
│ 对每个原始 token (blockIdx.x):                             │
│                                                              │
│  for k_idx in range(experts_per_token):  // top-k        │
│    1. 计算扩展索引:                                       │
│       expanded_original_row = token_idx + k_idx * num_rows │
│                                                             │
│    2. 从映射表查找排列后的位置:                           │
│       expanded_permuted_row =                             │
│         expanded_source_row_to_expanded_dest_row[        │
│           expanded_original_row                           │
│         ]                                                  │
│                                                             │
│    3. 获取该行的缩放因子:                                │
│       row_scale = scales[token_idx * top_k + k_idx]     │
│                                                             │
│    4. 从 GEMM2 输出读取该行数据:                         │
│       row_data = gemm_output[expanded_permuted_row, :]   │
│                                                             │
│    5. 应用缩放:                                           │
│       scaled_data = row_data * row_scale                 │
│                                                             │
│    6. 累加到原始位置:                                    │
│       final_output[token_idx, :] += scaled_data          │
│                                                             │
│  end for                                                   │
│                                                             │
│  7. 应用偏置 (仅 tp_rank=0):                             │
│     final_output[token_idx, :] += bias                   │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼ cudaDeviceSynchronize() [最终同步]
┌──────────────────────────────────────────────────────────────┐
│ Unpermuted Final Output                                      │
│                                                              │
│ final_output = [                                            │
│   Token0@ExpertA * s0_a + Token0@ExpertB * s0_b + bias,   │
│   Token1@ExpertA * s1_a + Token1@ExpertC * s1_c + bias,   │
│   Token2@ExpertB * s2_b + Token2@ExpertC * s2_c + bias,   │
│   ...                                                       │
│ ]                                                           │
│                                                              │
│ Shape: [num_rows, hidden_size]                            │
│ 顺序: 原始 token 顺序恢复                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 数据结构和映射表详解

### 关键映射表

#### 1. `expanded_source_row_to_expanded_dest_row`

**目的**: 将扩展源行映射到扩展目标行（unpermute 的反向）

```
┌─────────────────────────────────────────────────────────┐
│ 构建过程:                                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 原始数据结构:                                          │
│  num_rows = 3 (tokens)                               │
│  experts_per_token = 2 (top-k=2)                    │
│  num_experts = 3                                     │
│                                                         │
│ token_selected_experts (原始):                        │
│  Token 0: [Expert 0, Expert 1]                      │
│  Token 1: [Expert 0, Expert 2]                      │
│  Token 2: [Expert 1, Expert 2]                      │
│                                                         │
│ 展开索引 (expanded_original_row):                    │
│  Token0@Expert0: row 0 + 0*3 = 0                   │
│  Token0@Expert1: row 0 + 1*3 = 3                   │
│  Token1@Expert0: row 1 + 0*3 = 1                   │
│  Token1@Expert2: row 1 + 1*3 = 4                   │
│  Token2@Expert1: row 2 + 0*3 = 2                   │
│  Token2@Expert2: row 2 + 1*3 = 5                   │
│                                                         │
│ GEMM2 中排列后的顺序 (expanded_permuted_row):       │
│  按专家分组:                                         │
│  Expert 0: [Token0, Token1] → permuted rows [0, 1] │
│  Expert 1: [Token0, Token2] → permuted rows [2, 3] │
│  Expert 2: [Token1, Token2] → permuted rows [4, 5] │
│                                                         │
│ 映射表:                                              │
│  expanded_source_row_to_expanded_dest_row[0] = 0    │
│  expanded_source_row_to_expanded_dest_row[3] = 2    │
│  expanded_source_row_to_expanded_dest_row[1] = 1    │
│  expanded_source_row_to_expanded_dest_row[4] = 4    │
│  expanded_source_row_to_expanded_dest_row[2] = 3    │
│  expanded_source_row_to_expanded_dest_row[5] = 5    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**在 finalizeMoeRoutingKernel 中的使用**:

```cuda
for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
    int64_t expanded_original_row = original_row + k_idx * num_rows;
    //  ↑ 计算原始展开行索引
    
    int64_t expanded_permuted_row = 
        expanded_source_row_to_expanded_dest_row[expanded_original_row];
    //  ↑ 查找该行在 GEMM2 输出中的位置
    
    // 从排列后的位置读取数据
    auto const* permuted_row_ptr = 
        expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;
    
    // 使用该数据累加到最终输出
}
```

#### 2. `expert_for_source_row`

**目的**: 记录每个扩展行对应的专家 ID

```cpp
// 对上例:
expert_for_source_row[0] = 0  // Token0 @ Expert 0
expert_for_source_row[1] = 0  // Token1 @ Expert 0
expert_for_source_row[2] = 1  // Token2 @ Expert 1
expert_for_source_row[3] = 1  // Token0 @ Expert 1
expert_for_source_row[4] = 2  // Token1 @ Expert 2
expert_for_source_row[5] = 2  // Token2 @ Expert 2

// 在 kernel 中用于调试和检查
int64_t expert_idx = expert_for_source_row[k_offset];
```

#### 3. `token_final_scales` (unpermuted)

**目的**: 每个 token-expert 对的 router 缩放因子

```cpp
// Shape: [num_rows, experts_per_token]
token_final_scales = [
    [s0_expert0, s0_expert1],  // Token 0
    [s1_expert0, s1_expert2],  // Token 1
    [s2_expert1, s2_expert2],  // Token 2
]

// 在 finalize kernel 中:
for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
    int64_t k_offset = original_row * experts_per_token + k_idx;
    float row_scale = scales[k_offset];  // ◄──── 获取该 token-expert 的 scale
}
```

---

## 关键变量的大小和关系

```
num_rows = 3                    // 原始 token 数量
num_experts = 3                 // 总专家数
experts_per_token = 2           // top-k = 2
hidden_size = 4096              // 隐藏维度
inter_size = 10666              // 中间维度

expanded_num_rows = num_rows * experts_per_token = 6

┌─────────────────────────────────────────────────────────────┐
│ GEMM2 输入 (Intermediate)                                  │
├─────────────────────────────────────────────────────────────┤
│ Shape: [expanded_num_rows, inter_size]                    │
│      = [6, 10666]                                          │
│ Memory: 6 * 10666 * 2 bytes (FP16) = 256 KB              │
└─────────────────────────────────────────────────────────────┘

↓ GEMM2 计算

┌─────────────────────────────────────────────────────────────┐
│ GEMM2 输出 (Permuted)                                      │
├─────────────────────────────────────────────────────────────┤
│ Shape: [expanded_num_rows, hidden_size]                   │
│      = [6, 4096]                                           │
│ Memory: 6 * 4096 * 2 bytes (FP16) = 48 KB               │
│ 顺序: 按专家分组 (permuted)                                │
└─────────────────────────────────────────────────────────────┘

↓ finalizeMoeRoutingKernel

┌─────────────────────────────────────────────────────────────┐
│ 映射表大小                                                 │
├─────────────────────────────────────────────────────────────┤
│ expanded_source_row_to_expanded_dest_row: 6 * int = 24 B │
│ expert_for_source_row: 6 * int = 24 B                   │
│ token_final_scales: 6 * float = 24 B                    │
│ Total: 72 B (极小)                                        │
└─────────────────────────────────────────────────────────────┘

↓ unpermute + scale + reduce

┌─────────────────────────────────────────────────────────────┐
│ 最终输出 (Unpermuted)                                       │
├─────────────────────────────────────────────────────────────┤
│ Shape: [num_rows, hidden_size]                            │
│      = [3, 4096]                                           │
│ Memory: 3 * 4096 * 2 bytes (FP16) = 24 KB               │
│ 顺序: 原始 token 顺序                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## finalizeMoeRoutingKernel 的 Thread Block 分配

```cpp
// Launcher 代码
int64_t const blocks = num_rows;           // 3 个 blocks
int64_t const threads = FINALIZE_THREADS_PER_BLOCK;  // 通常 1024

// Grid 和 Block 配置
dim3 grid(blocks, 1, 1);           // grid = [3, 1, 1]
dim3 block(threads, 1, 1);         // block = [1024, 1, 1]

// ◄──── 关键: 每个 block 处理一个原始 token
```

### Thread Block 工作流

```
Block 0: 处理 Token 0
    ├─ K_idx=0: 读取 Token0@Expert0 的结果
    ├─ K_idx=1: 读取 Token0@Expert1 的结果
    └─ 生成: final_output[0, :]

Block 1: 处理 Token 1
    ├─ K_idx=0: 读取 Token1@Expert0 的结果
    ├─ K_idx=1: 读取 Token1@Expert2 的结果
    └─ 生成: final_output[1, :]

Block 2: 处理 Token 2
    ├─ K_idx=0: 读取 Token2@Expert1 的结果
    ├─ K_idx=1: 读取 Token2@Expert2 的结果
    └─ 生成: final_output[2, :]
```

### 单个 Thread Block 的计算（以 Block 0 为例）

```cuda
__global__ void finalizeMoeRoutingKernel(...) {
    int64_t original_row = blockIdx.x;  // = 0
    int64_t num_rows = gridDim.x;       // = 3
    int start_offset = threadIdx.x;     // [0, 1023]
    int stride = blockDim.x;            // = 1024
    
    // 每个 thread 处理 hidden_size/FINALIZE_ELEM_PER_THREAD 个元素
    // FINALIZE_ELEM_PER_THREAD = 128 bits / min(OutputType_bits, GemmOutputType_bits)
    //                           = 128 / 16 = 8 (对于 FP16)
    
    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        // elem_index 遍历 [0, 512) 中的元素
        // (4096 hidden_size / 8 elem_per_thread = 512)
        
        bool has_valid = false;
        ComputeElem thread_output = 0;  // 累加器
        
        // ============ K-Way Reduction ============
        for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {  // k=0,1
            
            // 计算扩展索引
            int64_t expanded_original_row = original_row + k_idx * num_rows;
            // k=0: 0 + 0*3 = 0
            // k=1: 0 + 1*3 = 3
            
            // 查找排列后的位置
            int64_t expanded_permuted_row = 
                expanded_source_row_to_expanded_dest_row[expanded_original_row];
            // k=0: [0] = 0  (Token0@Expert0 在 permuted 中的位置)
            // k=1: [3] = 2  (Token0@Expert1 在 permuted 中的位置)
            
            // 获取 scale
            int64_t k_offset = original_row * experts_per_token + k_idx;
            // k=0: 0*2 + 0 = 0
            // k=1: 0*2 + 1 = 1
            
            float row_scale = scales[k_offset];
            // k=0: scales[0] = s0_expert0
            // k=1: scales[1] = s0_expert1
            
            // 加载 GEMM2 输出
            auto const* permuted_row_ptr = 
                expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;
            
            auto permuted_val = permuted_row_ptr[elem_index];
            
            // 累加
            thread_output += permuted_val * row_scale;
        }
        
        // ============ 应用偏置 (仅 tp_rank=0) ============
        if (bias != nullptr) {
            auto bias_val = bias_v[original_row * num_elems_in_col + elem_index];
            thread_output += bias_val;
        }
        
        // ============ 写回最终输出 ============
        reduced_row_ptr_v[elem_index] = convert<OutputElem>(thread_output);
    }
}
```

---

## 内存访问模式分析

### GEMM2 输出的内存布局 (Permuted)

```
假设有 6 个扩展行，hidden_size = 4096，FINALIZE_ELEM_PER_THREAD = 8

内存布局（linear）:
[Row 0] [Row 1] [Row 2] [Row 3] [Row 4] [Row 5]
 ├─512──┼─512──┼─512──┼─512──┼─512──┼─512──┤
 └─4096─┘└─4096─┘└─4096─┘└─4096─┘└─4096─┘└─4096─┘

每个 row 的 4096 个元素被分成 512 个"chunks"（每个 8 个元素）
```

### finalizeMoeRoutingKernel 的访问模式

```
Block 0 处理 Token 0，线程分配:
  Thread 0:   elem_index = [0, 512, 1024, 1536, ...]
              ↓ 读取 permuted_row_ptr[expanded_permuted_row=0][elem_index]
              ↓ 读取 permuted_row_ptr[expanded_permuted_row=2][elem_index]
              
  Thread 1:   elem_index = [1, 513, 1025, 1537, ...]
              ↓ 同样的访问模式
              
  ...
  
  Thread 1023: elem_index = [1023]  (超出范围，跳过)

访问特征:
  - 主要特性: 随机访问多个 permuted rows
  - DRAM 行冲突: 可能频繁
  - L2 缓存命中: 取决于数据布局
  - 工作量不均衡: 元素较少时有些线程闲置
```

---

## 性能特性

### 计算强度 (Arithmetic Intensity)

```
finalizeMoeRoutingKernel 的操作:
  - 对每个 token 的每个 experts_per_token 迭代
  - 每次迭代: 1 次 load + 1 次 multiply + 1 次 add + 1 次 store
  - 总计: ~3-4 次 FLOPs 每个元素

数据量:
  - 输入: expanded_num_rows * hidden_size * 2 bytes
  - 映射表: ~3 * expanded_num_rows * 4 bytes (negligible)
  - 输出: num_rows * hidden_size * 2 bytes

总 FLOP:
  - num_rows * hidden_size * experts_per_token * 4 FLOPs

总内存访问:
  - ~num_rows * hidden_size * 2 bytes (主要是 GEMM2 输出的读取)

Arithmetic Intensity = (FLOP / Bytes) 相对较低
→ 内存绑定的 kernel
→ 应尽量增大缓存命中率和内存带宽利用率
```

### Occupancy 分析

```
SM80 (GA100) 特性:
  - 最大 thread blocks per SM: 32
  - Warp 大小: 32
  - 共享内存: 96 KB
  - 寄存器: 256 KB/SM

finalizeMoeRoutingKernel 配置:
  - blockSize = 1024 = 32 warps
  - gridSize = num_rows
  - sharedMem = 0 (不使用共享内存)
  - 寄存器需求: ~30-50 per thread
  
Occupancy = (32 warp * 32 thread/warp) / 2048 = ~50%
```

---

## SM80 vs SM90 的区别

```
┌────────────────────────────────────────────────────────┐
│           SM80 (Ampere)        │    SM90 (Hopper)      │
├────────────────────────────────────────────────────────┤
│ finalize kernel:  独立 kernel  │  融合在 epilogue      │
│ 调用次数:         总是 1 次    │  可能 0 次            │
│ kernel 同步:      显式         │  隐式 (通过 PDL)      │
│ TMA 支持:         ✗            │  ✓                    │
│ 内存访问:         随机         │  TMA 指令优化         │
└────────────────────────────────────────────────────────┘
```

---

## 总结

对于 **SM80 上的 finalizeMoeRoutingKernel**：

1. **何时调用**: GEMM2 完成后，通过 `finalizeMoeRoutingKernelLauncher` 启动
2. **何时跳过**: min_latency_mode=true （SM80 不支持）
3. **功能**:
   - Unpermute: 恢复原始 token 顺序
   - K-Way Reduction: 聚集多个专家的结果
   - Scale & Bias: 应用最终的缩放和偏置
4. **内存访问**: 对 GEMM2 输出的随机访问
5. **性能**: 内存绑定，occupancy ~50%
6. **必要性**: 关键步骤，生成最终正确的输出
