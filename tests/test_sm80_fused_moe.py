"""
Test SM80 (Ampere) support for cutlass_fused_moe API

This test verifies that the SM80 backend works correctly on Ampere GPUs (A100, A10, etc.)
"""

import torch
import pytest


def test_sm80_detection():
    """Test that SM80 architecture is correctly detected"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(device)
    
    print(f"Detected compute capability: SM{major}{minor}")
    
    # This test should run on SM80+ (Ampere or newer)
    assert major >= 8, f"SM80 tests require SM80+ architecture, got SM{major}{minor}"


def test_sm80_fused_moe_fp16():
    """Test SM80 fused MoE with FP16 activations and weights"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda:0")
    major, _ = torch.cuda.get_device_capability(device)
    
    if major < 8:
        pytest.skip("SM80 tests require SM80+ (Ampere or newer)")
    
    # Import after checking architecture
    from flashinfer import cutlass_fused_moe
    
    # Test parameters
    seq_len = 4
    hidden_size = 128
    intermediate_size = 256
    num_experts = 8
    top_k = 2
    
    # Create test tensors
    input = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device)
    token_selected_experts = torch.randint(0, num_experts, (seq_len, top_k), dtype=torch.int32, device=device)
    token_final_scales = torch.rand(seq_len, top_k, dtype=torch.float32, device=device)
    
    fc1_expert_weights = torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.float16, device=device)
    fc2_expert_weights = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.float16, device=device)
    
    # Empty quant_scales for non-quantized case
    quant_scales = []
    
    # Run fused MoE
    output = cutlass_fused_moe(
        input=input,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        fc1_expert_weights=fc1_expert_weights,
        fc2_expert_weights=fc2_expert_weights,
        output_dtype=torch.float16,
        quant_scales=quant_scales,
    )
    
    # Check output shape
    assert output.shape == (seq_len, hidden_size)
    assert output.dtype == torch.float16
    assert not torch.isnan(output).any()
    print(f"✓ SM80 FP16 test passed. Output shape: {output.shape}")


def test_sm80_fused_moe_bf16():
    """Test SM80 fused MoE with BF16 activations and weights"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda:0")
    major, _ = torch.cuda.get_device_capability(device)
    
    if major < 8:
        pytest.skip("SM80 tests require SM80+ (Ampere or newer)")
    
    from flashinfer import cutlass_fused_moe
    
    # Test parameters
    seq_len = 4
    hidden_size = 128
    intermediate_size = 256
    num_experts = 8
    top_k = 2
    
    # Create test tensors
    input = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    token_selected_experts = torch.randint(0, num_experts, (seq_len, top_k), dtype=torch.int32, device=device)
    token_final_scales = torch.rand(seq_len, top_k, dtype=torch.float32, device=device)
    
    fc1_expert_weights = torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)
    fc2_expert_weights = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
    
    quant_scales = []
    
    output = cutlass_fused_moe(
        input=input,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        fc1_expert_weights=fc1_expert_weights,
        fc2_expert_weights=fc2_expert_weights,
        output_dtype=torch.bfloat16,
        quant_scales=quant_scales,
    )
    
    assert output.shape == (seq_len, hidden_size)
    assert output.dtype == torch.bfloat16
    assert not torch.isnan(output).any()
    print(f"✓ SM80 BF16 test passed. Output shape: {output.shape}")


def test_sm80_unsupported_fp8():
    """Test that SM80 correctly rejects FP8 inputs"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda:0")
    major, _ = torch.cuda.get_device_capability(device)
    
    if major < 8:
        pytest.skip("SM80 tests require SM80+ (Ampere or newer)")
    
    if major >= 9:
        pytest.skip("This test is for SM80 limitations, skipping on SM90+")
    
    from flashinfer import cutlass_fused_moe
    
    # Test parameters
    seq_len = 4
    hidden_size = 128
    intermediate_size = 256
    num_experts = 8
    top_k = 2
    
    # Try to use FP8 (should fail on SM80)
    input = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    token_selected_experts = torch.randint(0, num_experts, (seq_len, top_k), dtype=torch.int32, device=device)
    token_final_scales = torch.rand(seq_len, top_k, dtype=torch.float32, device=device)
    
    fc1_expert_weights = torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.float8_e4m3fn, device=device)
    fc2_expert_weights = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.float8_e4m3fn, device=device)
    
    quant_scales = []
    
    # This should raise NotImplementedError on SM80
    with pytest.raises(NotImplementedError, match="FP8"):
        output = cutlass_fused_moe(
            input=input,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            fc1_expert_weights=fc1_expert_weights,
            fc2_expert_weights=fc2_expert_weights,
            output_dtype=torch.float16,
            quant_scales=quant_scales,
        )
    
    print("✓ SM80 correctly rejects FP8 inputs")


def test_sm80_unsupported_min_latency_mode():
    """Test that SM80 correctly rejects min_latency_mode"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda:0")
    major, _ = torch.cuda.get_device_capability(device)
    
    if major < 8:
        pytest.skip("SM80 tests require SM80+ (Ampere or newer)")
    
    if major >= 9:
        pytest.skip("This test is for SM80 limitations, skipping on SM90+")
    
    from flashinfer import cutlass_fused_moe
    
    # Test parameters
    seq_len = 4
    hidden_size = 128
    intermediate_size = 256
    num_experts = 8
    top_k = 2
    
    input = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device)
    token_selected_experts = torch.randint(0, num_experts, (seq_len, top_k), dtype=torch.int32, device=device)
    token_final_scales = torch.rand(seq_len, top_k, dtype=torch.float32, device=device)
    
    fc1_expert_weights = torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.float16, device=device)
    fc2_expert_weights = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.float16, device=device)
    
    quant_scales = []
    
    # This should raise NotImplementedError on SM80
    with pytest.raises(NotImplementedError, match="Min latency mode"):
        output = cutlass_fused_moe(
            input=input,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            fc1_expert_weights=fc1_expert_weights,
            fc2_expert_weights=fc2_expert_weights,
            output_dtype=torch.float16,
            quant_scales=quant_scales,
            min_latency_mode=True,
        )
    
    print("✓ SM80 correctly rejects min_latency_mode")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing SM80 (Ampere) support for cutlass_fused_moe")
    print("=" * 70)
    
    test_sm80_detection()
    print()
    
    try:
        test_sm80_fused_moe_fp16()
    except Exception as e:
        print(f"✗ SM80 FP16 test failed: {e}")
    
    try:
        test_sm80_fused_moe_bf16()
    except Exception as e:
        print(f"✗ SM80 BF16 test failed: {e}")
    
    try:
        test_sm80_unsupported_fp8()
    except Exception as e:
        print(f"✗ SM80 FP8 rejection test failed: {e}")
    
    try:
        test_sm80_unsupported_min_latency_mode()
    except Exception as e:
        print(f"✗ SM80 min_latency_mode rejection test failed: {e}")
    
    print()
    print("=" * 70)
    print("All SM80 tests completed!")
    print("=" * 70)
