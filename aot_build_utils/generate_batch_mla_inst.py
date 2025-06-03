"""
Copyright (c) 2024 by FlashInfer team.

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

import re
import sys
from pathlib import Path

from .literal_map import dtype_literal, idtype_literal


def get_cu_file_str(
    head_dim_ckv,
    head_dim_kpe,
    dtype_q,
    dtype_kv,
    dtype_out,
    idtype,
):
    content = """#include <flashinfer/attention/mla.cuh>
#include <flashinfer/attention/scheduler.cuh>

namespace flashinfer {{

using Params = MLAParams<{dtype_q}, {dtype_kv}, {dtype_out}, {idtype}>;

template cudaError_t mla::BatchMLAPagedAttention<MaskMode::kNone, {head_dim_ckv}, {head_dim_kpe}, Params>(
    Params params,
    uint32_t num_blks_x,
    uint32_t num_blks_y,
    cudaStream_t stream);

template cudaError_t mla::BatchMLAPagedAttention<MaskMode::kCausal, {head_dim_ckv}, {head_dim_kpe}, Params>(
    Params params,
    uint32_t num_blks_x,
    uint32_t num_blks_y,
    cudaStream_t stream);

template cudaError_t MLAPlan<{idtype}>(
    void* float_buffer,
    size_t float_workspace_size_in_bytes,
    void* int_buffer,
    void* page_locked_int_buffer,
    size_t int_workspace_size_in_bytes,
    MLAPlanInfo& plan_info,
    {idtype}* qo_indptr_h,
    {idtype}* kv_indptr_h,
    {idtype}* kv_len_arr_h,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t head_dim_o,
    bool causal,
    cudaStream_t stream);

}}
    """.format(
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        dtype_q=dtype_literal[dtype_q],
        dtype_kv=dtype_literal[dtype_kv],
        dtype_out=dtype_literal[dtype_out],
        idtype=idtype_literal[idtype],
    )
    return content


if __name__ == "__main__":
    pattern = (
        r"batch_mla_head_ckv_([0-9]+)_head_kpe_([0-9]+)_"
        r"dtypeq_([a-z0-9]+)_dtypekv_([a-z0-9]+)_dtypeout_([a-z0-9]+)_idtype_([a-z0-9]+)\.cu"
    )

    compiled_pattern = re.compile(pattern)
    path = Path(sys.argv[1])
    fname = path.name
    match = compiled_pattern.match(fname)
    with open(path, "w") as f:
        f.write(get_cu_file_str(*match.groups())) 