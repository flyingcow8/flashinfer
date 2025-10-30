#!/usr/bin/env python3
"""验证 SM80 fused_moe 是否正确添加到 AOT 编译中"""

import sys

def check_aot_integration():
    """检查 aot.py 中的 SM80 集成"""
    
    print("🔍 检查 AOT 编译集成...")
    print("=" * 70)
    
    # 1. 检查 import
    with open('flashinfer/aot.py', 'r') as f:
        content = f.read()
    
    checks = [
        ('import gen_fused_moe_sm80_module', 
         'gen_fused_moe_sm80_module' in content and 
         'from .fused_moe import gen_fused_moe_sm80_module' in content),
        
        ('gen_fused_moe_sm80_module() 调用', 
         'jit_specs.append(gen_fused_moe_sm80_module())' in content),
        
        ('SM80 模块在 SM100 条件之外', 
         content.index('gen_fused_moe_sm80_module()') < content.index('if has_sm100:')),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("✅ 所有检查通过！")
        print()
        print("📝 修改摘要:")
        print("  1. 导入: from .fused_moe import gen_fused_moe_sm80_module")
        print("  2. 添加: jit_specs.append(gen_fused_moe_sm80_module())")
        print("  3. 位置: SM80 模块在主流程中编译（适用于所有架构）")
        print()
        print("�� 下一步:")
        print("  - 运行 AOT 编译: python -m flashinfer.aot --out-dir aot-ops")
        print("  - 验证生成的模块: ls aot-ops/fused_moe_sm80/")
        return 0
    else:
        print("❌ 部分检查失败")
        return 1

if __name__ == "__main__":
    sys.exit(check_aot_integration())
