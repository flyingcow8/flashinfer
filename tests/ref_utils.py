"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from typing import List, Optional
import math
from einops import repeat


def get_paged_kv(
    data,
    start_idx,
    end_idx,
    last_len,
    kv_layout,
    perm_dims,
    perm_dims_last,
    num_kv_heads,
    head_dim,
):
    """Helper function to get paged key/value tensors.

    Args:
        data: Input tensor (k_data or v_data)
        start_idx: Start index for the current batch
        end_idx: End index for the current batch
        last_len: Length of the last page
        kv_layout: KV layout format ("HND" or "NHD")
        perm_dims: Permutation dimensions for full pages
        perm_dims_last: Permutation dimensions for last page
        num_kv_heads: Number of KV heads
        head_dim: Dimension of each head
    """
    # Handle full pages
    full = (
        data[start_idx : end_idx - 1]
        .permute(*perm_dims)
        .reshape(-1, num_kv_heads, head_dim)
    )

    # Handle last partial page
    last_page_idx = end_idx - 1
    if kv_layout == "HND":
        last = data[last_page_idx, :, :last_len]
    else:  # NHD
        last = data[last_page_idx, :last_len, :]
    last = last.permute(*perm_dims_last).reshape(-1, num_kv_heads, head_dim)

    # Concatenate full and partial pages
    return torch.cat([full, last], dim=0)


def get_ragged_kv(data, start_idx, end_idx, perm_dims_last, num_kv_heads, head_dim):
    """Helper function to get ragged key/value tensors.

    Args:
        data: Input tensor (k_data or v_data)
        start_idx: Start index for the current batch
        end_idx: End index for the current batch
        perm_dims_last: Permutation dimensions
        num_kv_heads: Number of KV heads
        head_dim: Dimension of each head
    """
    return (
        data[start_idx:end_idx]
        .permute(*perm_dims_last)
        .reshape(-1, num_kv_heads, head_dim)
    )


def attn_batch_ref(
    q: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    q_indptr=None,
    kv_indptr=None,
    kv_last_page_len=None,  # None for ragged attention, array for paged attention
    qo_len=None,
    kv_len=None,
    num_kv_heads=None,
    kv_layout=None,
    causal: Optional[bool] = False,
    logits_soft_cap: Optional[float] = None,
    upcast=False,
):
    """Reference implementation for both paged and ragged attention.

    Args:
        q: Query tensor [total_q_len/batch, num_heads, head_dim]
        k_data: Key tensor [total_kv_len/batch, num_kv_heads/layout_dim, head_dim/layout_dim]
        v_data: Value tensor [total_kv_len/batch, num_kv_heads/layout_dim, head_dim/layout_dim]
        q_indptr: Query index pointer [batch_size + 1] for ragged format
        kv_indptr: KV cache index pointer [batch_size + 1]
        kv_last_page_len: Last page lengths for paged format [batch_size]
        qo_len: Query length for paged format
        kv_len: KV length for paged format
        num_kv_heads: Number of KV heads
        kv_layout: KV layout format ("HND" or "NHD") for paged format
        causal: Whether to apply causal mask
        logits_soft_cap: Soft cap for attention logits
        upcast: Whether to upcast to float32
    """
    batch_size = kv_indptr.shape[0] - 1
    assert kv_layout in ["HND", "NHD"], f"The parameter kv_layout is invalid!!!"

    logits_soft_cap = logits_soft_cap or 0.0
    perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
    perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
    head_dim = q.shape[-1]

    # Handle precision
    dtype = torch.float if upcast else torch.half
    q = q.to(dtype)
    k_data = k_data.to(dtype)
    v_data = v_data.to(dtype)

    is_paged = kv_last_page_len is not None
    output: List[torch.Tensor] = []
    for i in range(batch_size):
        # Handle query tensor
        qi = q[q_indptr[i] : q_indptr[i + 1]] if q_indptr is not None else q[i]
        if qi.dim() == 2:
            qi = qi.unsqueeze(0)

        # Handle key and value tensors based on attention type
        if is_paged:  # Paged attention
            ki = get_paged_kv(
                k_data,
                kv_indptr[i],
                kv_indptr[i + 1],
                kv_last_page_len[i],
                kv_layout,
                perm_dims,
                perm_dims_last,
                num_kv_heads,
                head_dim,
            )
            vi = get_paged_kv(
                v_data,
                kv_indptr[i],
                kv_indptr[i + 1],
                kv_last_page_len[i],
                kv_layout,
                perm_dims,
                perm_dims_last,
                num_kv_heads,
                head_dim,
            )
        else:  # Ragged attention
            ki = get_ragged_kv(
                k_data,
                kv_indptr[i],
                kv_indptr[i + 1],
                perm_dims_last,
                num_kv_heads,
                head_dim,
            )
            vi = get_ragged_kv(
                v_data,
                kv_indptr[i],
                kv_indptr[i + 1],
                perm_dims_last,
                num_kv_heads,
                head_dim,
            )

        # Repeat for multi-query attention if needed
        ki = repeat(ki, "t h d -> t (h g) d", g=qi.shape[1] // ki.shape[1])
        vi = repeat(vi, "t h d -> t (h g) d", g=qi.shape[1] // vi.shape[1])

        # Compute attention
        attn = torch.einsum("qhd,khd->hqk", qi / math.sqrt(head_dim), ki).float()

        # Apply causal mask if needed
        if causal:
            empty_mask = torch.ones(qo_len, kv_len)
            diag_ = abs(kv_len - qo_len)
            mask = torch.triu(empty_mask, diagonal=diag_ + 1).bool().to(0)
            attn.masked_fill_(mask, float("-inf"))

        # Apply soft cap if specified
        if logits_soft_cap > 0:
            attn = logits_soft_cap * torch.tanh(attn / logits_soft_cap)

        # Compute output
        attn = torch.softmax(attn, dim=-1).to(vi.dtype)
        o_ref_torch = torch.einsum("hqk,khd->qhd", attn, vi)
        output.append(o_ref_torch)

    return torch.cat(output, dim=0)
