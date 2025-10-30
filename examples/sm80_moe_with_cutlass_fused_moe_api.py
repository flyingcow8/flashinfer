"""
Example: Using simplified SM80 MoE with cutlass_fused_moe API

This example shows how to use the simplified SM80 MoE implementation
that's integrated into the existing cutlass_fused_moe API.

The simplified implementation removes dependencies on TensorRT-LLM and CUTLASS,
but requires you to provide GEMM implementations.

Requirements:
- SM80+ GPU (Ampere or newer)
- PyTorch with CUDA support
- flashinfer package

Usage:
    python examples/sm80_moe_with_cutlass_fused_moe_api.py
"""

import torch
import flashinfer

def create_simple_gemm_implementation():
    """
    Create GEMM implementations using PyTorch's native ops.
    
    In production, you would replace these with optimized CUDA kernels
    (e.g., cuBLAS, CUTLASS, or custom implementations).
    """
    
    # GEMM1: Grouped GEMM for first layer (hidden_size -> intermediate_size)
    # Shape: [total_tokens, hidden_size] @ [expert_weights: inter_size, hidden_size]^T
    def gemm1_impl(
        permuted_input,      # [total_expanded_tokens, hidden_size]
        fc1_weights,         # [num_experts, inter_size, hidden_size]
        expert_first_token_offset,  # [num_experts + 1]
        num_experts
    ):
        """
        Perform grouped GEMM for all experts in parallel.
        Each expert processes a different subset of tokens.
        """
        outputs = []
        for expert_id in range(num_experts):
            start_idx = expert_first_token_offset[expert_id].item()
            end_idx = expert_first_token_offset[expert_id + 1].item()
            
            if end_idx > start_idx:
                # Extract tokens for this expert
                expert_input = permuted_input[start_idx:end_idx]  # [num_tokens_this_expert, hidden]
                expert_weight = fc1_weights[expert_id]  # [inter_size, hidden]
                
                # GEMM: [num_tokens, hidden] @ [hidden, inter_size]
                expert_output = torch.matmul(expert_input, expert_weight.t())
                outputs.append(expert_output)
        
        # Concatenate all expert outputs
        if outputs:
            return torch.cat(outputs, dim=0)  # [total_tokens, inter_size]
        else:
            return torch.empty((0, fc1_weights.shape[1]), 
                             dtype=permuted_input.dtype, 
                             device=permuted_input.device)
    
    # GEMM2: Grouped GEMM for second layer (intermediate_size -> hidden_size)
    def gemm2_impl(
        activation_output,   # [total_expanded_tokens, intermediate_size]
        fc2_weights,         # [num_experts, hidden_size, inter_size]
        expert_first_token_offset,  # [num_experts + 1]
        num_experts
    ):
        """
        Perform grouped GEMM for all experts in parallel.
        """
        outputs = []
        for expert_id in range(num_experts):
            start_idx = expert_first_token_offset[expert_id].item()
            end_idx = expert_first_token_offset[expert_id + 1].item()
            
            if end_idx > start_idx:
                expert_input = activation_output[start_idx:end_idx]
                expert_weight = fc2_weights[expert_id]  # [hidden_size, inter_size]
                
                # GEMM: [num_tokens, inter] @ [inter, hidden]
                expert_output = torch.matmul(expert_input, expert_weight.t())
                outputs.append(expert_output)
        
        if outputs:
            return torch.cat(outputs, dim=0)  # [total_tokens, hidden_size]
        else:
            return torch.empty((0, fc2_weights.shape[1]), 
                             dtype=activation_output.dtype, 
                             device=activation_output.device)
    
    return gemm1_impl, gemm2_impl


def example_fp16_moe():
    """Example: FP16 MoE using cutlass_fused_moe API on SM80"""
    
    print("=" * 80)
    print("Example: FP16 MoE with cutlass_fused_moe API (SM80 simplified backend)")
    print("=" * 80)
    
    # Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA not available, skipping example")
        return
    
    # Model configuration
    batch_size = 4
    seq_len = 32
    num_tokens = batch_size * seq_len
    hidden_size = 512
    intermediate_size = 2048
    num_experts = 8
    top_k = 2
    
    # Create input tensors
    input_tensor = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    
    # Create expert weights (Swiglu requires 2x intermediate size)
    fc1_weights = torch.randn(num_experts, 2 * intermediate_size, hidden_size, 
                              dtype=torch.float16, device=device)
    fc2_weights = torch.randn(num_experts, hidden_size, intermediate_size,
                              dtype=torch.float16, device=device)
    
    # Create routing information (simulate router output)
    # token_selected_experts: [num_tokens, top_k] with expert indices
    token_selected_experts = torch.randint(0, num_experts, (num_tokens, top_k), 
                                           dtype=torch.int32, device=device)
    
    # token_final_scales: [num_tokens, top_k] with routing weights (should sum to 1 per token)
    token_final_scales = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
    
    # Quantization scales (empty for FP16)
    quant_scales = []
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"FC1 weights shape: {fc1_weights.shape}")
    print(f"FC2 weights shape: {fc2_weights.shape}")
    print(f"Selected experts shape: {token_selected_experts.shape}")
    print(f"Token scales shape: {token_final_scales.shape}")
    
    try:
        # Call the unified API - will automatically use SM80 simplified backend
        output = flashinfer.cutlass_fused_moe(
            input=input_tensor,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            fc1_expert_weights=fc1_weights,
            fc2_expert_weights=fc2_weights,
            output_dtype=torch.float16,
            quant_scales=quant_scales,
            tp_size=1,  # No tensor parallelism in simplified implementation
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
        )
        
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print("✅ MoE forward pass succeeded!")
        
        # Verify output is not NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        print("✅ Output validation passed!")
        
    except RuntimeError as e:
        if "GEMM" in str(e):
            print("\n⚠️  GEMM implementation not found!")
            print("The simplified SM80 backend requires you to provide GEMM implementations.")
            print("Please implement simple_moe_gemm1() and simple_moe_gemm2() functions,")
            print("or wait for the full integration with cuBLAS/CUTLASS.")
        else:
            raise


def example_int8_moe():
    """Example: INT8 quantized MoE"""
    
    print("\n" + "=" * 80)
    print("Example: INT8 Quantized MoE with cutlass_fused_moe API (SM80)")
    print("=" * 80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA not available, skipping example")
        return
    
    # Configuration
    num_tokens = 128
    hidden_size = 512
    intermediate_size = 2048
    num_experts = 8
    top_k = 2
    
    # Create FP16 tensors first
    input_fp16 = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    fc1_weights_fp16 = torch.randn(num_experts, 2 * intermediate_size, hidden_size,
                                   dtype=torch.float16, device=device)
    fc2_weights_fp16 = torch.randn(num_experts, hidden_size, intermediate_size,
                                   dtype=torch.float16, device=device)
    
    # Quantize to INT8 (simple quantization for demo)
    input_scale = input_fp16.abs().max() / 127.0
    input_int8 = (input_fp16 / input_scale).round().clamp(-128, 127).to(torch.int8)
    
    fc1_scale = fc1_weights_fp16.abs().max() / 127.0
    fc1_weights_int8 = (fc1_weights_fp16 / fc1_scale).round().clamp(-128, 127).to(torch.int8)
    
    fc2_scale = fc2_weights_fp16.abs().max() / 127.0
    fc2_weights_int8 = (fc2_weights_fp16 / fc2_scale).round().clamp(-128, 127).to(torch.int8)
    
    # Routing info
    token_selected_experts = torch.randint(0, num_experts, (num_tokens, top_k),
                                          dtype=torch.int32, device=device)
    token_final_scales = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
    
    # Quantization scales for INT8
    quant_scales = [
        input_scale.view(1),
        fc1_scale.view(1),
        fc2_scale.view(1),
        torch.tensor([1.0], device=device),  # Output scale
    ]
    
    print(f"Input shape: {input_int8.shape}, dtype: {input_int8.dtype}")
    print(f"Input quantization scale: {input_scale.item():.6f}")
    
    try:
        output = flashinfer.cutlass_fused_moe(
            input=input_int8,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            fc1_expert_weights=fc1_weights_int8,
            fc2_expert_weights=fc2_weights_int8,
            output_dtype=torch.float16,
            quant_scales=quant_scales,
        )
        
        print(f"Output shape: {output.shape}, dtype: {output.dtype}")
        print("✅ INT8 MoE forward pass succeeded!")
        
    except RuntimeError as e:
        if "GEMM" in str(e):
            print("\n⚠️  GEMM implementation not found!")
            print("INT8 MoE requires custom GEMM kernels with quantization support.")
        else:
            raise


def compare_with_reference():
    """Compare simplified implementation with reference (when available)"""
    
    print("\n" + "=" * 80)
    print("Comparison: Simplified SM80 vs Reference Implementation")
    print("=" * 80)
    
    print("This comparison would require a reference implementation.")
    print("Once you provide GEMM implementations, you can benchmark against:")
    print("  - PyTorch native MoE (using torch.einsum)")
    print("  - Original CUTLASS implementation (if available)")
    print("  - Custom CUDA kernels")


if __name__ == "__main__":
    print("FlashInfer Simplified SM80 MoE Examples")
    print("========================================\n")
    
    # Check GPU compute capability
    if torch.cuda.is_available():
        device_prop = torch.cuda.get_device_properties(0)
        compute_capability = device_prop.major * 10 + device_prop.minor
        print(f"GPU: {device_prop.name}")
        print(f"Compute Capability: SM{device_prop.major}{device_prop.minor}")
        
        if device_prop.major < 8:
            print("\n⚠️  Warning: This example requires SM80+ (Ampere or newer)")
            print("Your GPU is older and may not be supported.")
        print()
    
    # Run examples
    example_fp16_moe()
    example_int8_moe()
    compare_with_reference()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Implement simple_moe_gemm1() and simple_moe_gemm2() functions")
    print("2. Compile the extension: python -c \"import flashinfer\"")
    print("3. Run performance benchmarks")
    print("4. Compare with CUTLASS implementation (if available)")
