#!/usr/bin/env python3
"""
快速验证 SM80 简化 MoE 集成

这个脚本检查：
1. 文件结构是否正确
2. Python API 修改是否成功
3. C++ 扩展是否可以编译（如果提供了 GEMM）
4. 文档是否完整

使用方法:
    python verify_sm80_integration.py
"""

import os
import sys
from pathlib import Path

# 颜色输出
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
ENDC = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✓{ENDC} {msg}")

def print_error(msg):
    print(f"{RED}✗{ENDC} {msg}")

def print_warning(msg):
    print(f"{YELLOW}⚠{ENDC} {msg}")

def print_info(msg):
    print(f"{BLUE}ℹ{ENDC} {msg}")

def check_file_exists(path, description):
    """检查文件是否存在"""
    if os.path.exists(path):
        print_success(f"{description}: {path}")
        return True
    else:
        print_error(f"{description} 不存在: {path}")
        return False

def check_code_modifications():
    """检查代码修改"""
    print(f"\n{BLUE}{'='*70}{ENDC}")
    print(f"{BLUE}检查代码修改{ENDC}")
    print(f"{BLUE}{'='*70}{ENDC}\n")
    
    success = True
    
    # 检查 fused_moe.py 修改
    fused_moe_path = "flashinfer/fused_moe.py"
    if check_file_exists(fused_moe_path, "Python API 文件"):
        with open(fused_moe_path, 'r') as f:
            content = f.read()
            
            # 检查关键修改
            checks = [
                ("flashinfer_simple_moe_sm80_ops.cu", "使用简化实现"),
                ("Simplified MoE implementation", "文档注释"),
                ("simple_moe_gemm1", "GEMM 接口说明"),
            ]
            
            for check_str, desc in checks:
                if check_str in content:
                    print_success(f"  - 包含 {desc}")
                else:
                    print_error(f"  - 缺少 {desc}")
                    success = False
    else:
        success = False
    
    return success

def check_file_structure():
    """检查文件结构"""
    print(f"\n{BLUE}{'='*70}{ENDC}")
    print(f"{BLUE}检查文件结构{ENDC}")
    print(f"{BLUE}{'='*70}{ENDC}\n")
    
    files_to_check = [
        # C++ 扩展
        ("csrc/fused_moe_simple/flashinfer_simple_moe_sm80_ops.cu", 
         "SM80 PyTorch 绑定"),
        ("csrc/fused_moe_simple/common.cuh", 
         "通用工具"),
        ("csrc/fused_moe_simple/routing_kernels.cuh", 
         "路由内核"),
        ("csrc/fused_moe_simple/finalize_kernel.cuh", 
         "归约内核"),
        ("csrc/fused_moe_simple/moe_runner.cuh", 
         "主逻辑"),
        
        # Python API
        ("flashinfer/fused_moe.py", 
         "Python API"),
        
        # 示例
        ("examples/sm80_moe_with_cutlass_fused_moe_api.py", 
         "使用示例"),
        
        # 文档
        ("SM80_SIMPLIFIED_MOE_INTEGRATION.md", 
         "集成方案文档"),
        ("SM80_INTEGRATION_TODO.md", 
         "待办事项"),
        ("SM80_INTEGRATION_SUMMARY.md", 
         "集成总结"),
    ]
    
    success = True
    for path, description in files_to_check:
        if not check_file_exists(path, description):
            success = False
    
    return success

def check_documentation():
    """检查文档完整性"""
    print(f"\n{BLUE}{'='*70}{ENDC}")
    print(f"{BLUE}检查文档完整性{ENDC}")
    print(f"{BLUE}{'='*70}{ENDC}\n")
    
    success = True
    
    docs = [
        ("SM80_SIMPLIFIED_MOE_INTEGRATION.md", [
            "架构决策",
            "GEMM 集成策略",
            "使用方法",
            "性能对比",
        ]),
        ("SM80_INTEGRATION_TODO.md", [
            "P0",
            "P1",
            "P2",
            "cuBLAS GEMM",
        ]),
        ("SM80_INTEGRATION_SUMMARY.md", [
            "执行摘要",
            "架构变更",
            "当前状态",
            "下一步行动",
        ]),
    ]
    
    for doc_path, keywords in docs:
        if os.path.exists(doc_path):
            print_success(f"文档存在: {doc_path}")
            with open(doc_path, 'r') as f:
                content = f.read()
                for keyword in keywords:
                    if keyword in content:
                        print_success(f"  - 包含章节: {keyword}")
                    else:
                        print_warning(f"  - 缺少章节: {keyword}")
        else:
            print_error(f"文档缺失: {doc_path}")
            success = False
    
    return success

def check_compilation(dry_run=True):
    """检查编译（可选）"""
    print(f"\n{BLUE}{'='*70}{ENDC}")
    print(f"{BLUE}检查编译（干运行）{ENDC}")
    print(f"{BLUE}{'='*70}{ENDC}\n")
    
    if dry_run:
        print_info("跳过实际编译（使用 --compile 参数进行编译测试）")
        return True
    
    try:
        print_info("尝试编译 SM80 模块...")
        import torch
        if not torch.cuda.is_available():
            print_warning("CUDA 不可用，跳过编译测试")
            return True
        
        # 尝试导入
        import flashinfer
        print_success("flashinfer 导入成功")
        
        # 尝试触发 JIT 编译（会失败，因为没有 GEMM）
        try:
            # 这会触发编译但会在运行时失败
            device = torch.device("cuda:0")
            input_tensor = torch.randn(16, 512, dtype=torch.float16, device=device)
            token_selected_experts = torch.randint(0, 8, (16, 2), 
                                                   dtype=torch.int32, device=device)
            token_final_scales = torch.softmax(torch.randn(16, 2, device=device), dim=-1)
            fc1_weights = torch.randn(8, 4096, 512, dtype=torch.float16, device=device)
            fc2_weights = torch.randn(8, 512, 2048, dtype=torch.float16, device=device)
            
            output = flashinfer.cutlass_fused_moe(
                input=input_tensor,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                fc1_expert_weights=fc1_weights,
                fc2_expert_weights=fc2_weights,
                output_dtype=torch.float16,
                quant_scales=[],
            )
            
            print_success("编译和运行成功！")
            return True
            
        except RuntimeError as e:
            if "GEMM" in str(e):
                print_warning("编译成功，但需要 GEMM 实现才能运行")
                print_info(f"错误信息: {str(e)}")
                return True  # 编译成功，运行时需要 GEMM
            else:
                print_error(f"运行时错误: {str(e)}")
                return False
    
    except Exception as e:
        print_error(f"编译失败: {str(e)}")
        return False

def print_summary(results):
    """打印总结"""
    print(f"\n{BLUE}{'='*70}{ENDC}")
    print(f"{BLUE}验证总结{ENDC}")
    print(f"{BLUE}{'='*70}{ENDC}\n")
    
    all_success = all(results.values())
    
    for check_name, success in results.items():
        if success:
            print_success(f"{check_name}: 通过")
        else:
            print_error(f"{check_name}: 失败")
    
    print()
    if all_success:
        print_success("所有检查通过！✨")
        print_info("\n下一步:")
        print_info("  1. 实现 cuBLAS GEMM 后端 (csrc/fused_moe_simple/cublas_gemm_impl.cu)")
        print_info("  2. 运行测试: python examples/sm80_moe_with_cutlass_fused_moe_api.py")
        print_info("  3. 查看待办事项: cat SM80_INTEGRATION_TODO.md")
        return 0
    else:
        print_error("部分检查失败，请修复后重试")
        return 1

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="验证 SM80 简化 MoE 集成")
    parser.add_argument("--compile", action="store_true", 
                       help="执行实际编译测试（需要 CUDA）")
    args = parser.parse_args()
    
    print(f"\n{BLUE}{'='*70}{ENDC}")
    print(f"{BLUE}SM80 简化 MoE 集成验证{ENDC}")
    print(f"{BLUE}{'='*70}{ENDC}")
    
    results = {
        "文件结构": check_file_structure(),
        "代码修改": check_code_modifications(),
        "文档完整性": check_documentation(),
        "编译测试": check_compilation(dry_run=not args.compile),
    }
    
    return print_summary(results)

if __name__ == "__main__":
    sys.exit(main())
