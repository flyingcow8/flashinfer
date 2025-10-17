#!/usr/bin/env python3
"""
快速测试 SM80 MOE 编译和基本功能
"""
import torch
import sys

print("=" * 70)
print("SM80 MOE 编译测试")
print("=" * 70)

# 检查 GPU
if not torch.cuda.is_available():
    print("❌ CUDA 不可用")
    sys.exit(1)

device = torch.device("cuda:0")
major, minor = torch.cuda.get_device_capability(device)
print(f"✓ 检测到 GPU: SM{major}{minor}")

if major < 8:
    print(f"❌ 需要 SM80+，当前是 SM{major}{minor}")
    sys.exit(1)

# 尝试导入和编译
print("\n尝试导入 flashinfer...")
try:
    import flashinfer
    print("✓ flashinfer 导入成功")
except Exception as e:
    print(f"❌ flashinfer 导入失败: {e}")
    sys.exit(1)

# 尝试加载 SM80 模块
print("\n尝试编译 SM80 MOE 模块...")
try:
    from flashinfer.fused_moe import gen_fused_moe_sm80_module
    module_spec = gen_fused_moe_sm80_module()
    print(f"✓ 生成 JitSpec: {module_spec.name}")
    print(f"  - 源文件数: {len(module_spec.sources)}")
    print("  - Sources:")
    for src in module_spec.sources:
        print(f"    - {src}")
    # 尝试编译（这会触发实际的编译）
    print("\n开始编译（这可能需要几分钟）...")
    compiled_module = module_spec.build_and_load(class_name="FusedMoeRunner")
    print("✓ SM80 MOE 模块编译成功!")
    
except Exception as e:
    print(f"❌ SM80 MOE 模块编译失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 简单的功能测试
print("\n运行简单功能测试...")
try:
    # 测试参数
    seq_len = 4
    hidden_size = 128
    intermediate_size = 256
    num_experts = 8
    top_k = 2
    
    # 创建测试张量 (FP16)
    input_tensor = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device)
    token_selected_experts = torch.randint(0, num_experts, (seq_len, top_k), dtype=torch.int32, device=device)
    token_final_scales = torch.rand(seq_len, top_k, dtype=torch.float32, device=device)
    
    fc1_expert_weights = torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.float16, device=device)
    fc2_expert_weights = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.float16, device=device)
    
    print(f"✓ 创建测试张量")
    print(f"  - Input shape: {input_tensor.shape}")
    print(f"  - FC1 weights shape: {fc1_expert_weights.shape}")
    print(f"  - FC2 weights shape: {fc2_expert_weights.shape}")
    
    # 尝试调用 cutlass_fused_moe
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
    
    print(f"✓ MOE 前向传播成功")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output dtype: {output.dtype}")
    print(f"  - Has NaN: {torch.isnan(output).any()}")
    
    if output.shape != (seq_len, hidden_size):
        print(f"❌ 输出形状错误: 期望 {(seq_len, hidden_size)}, 得到 {output.shape}")
        sys.exit(1)
    
    if torch.isnan(output).any():
        print("❌ 输出包含 NaN")
        sys.exit(1)
    
    print("\n✓ 所有测试通过!")
    
except Exception as e:
    print(f"❌ 功能测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("SM80 MOE 测试完成!")
print("=" * 70)
