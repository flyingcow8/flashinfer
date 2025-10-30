```instructions
# FlashInfer AI Agent Guide

## Architecture Overview
**FlashInfer** is a CUDA kernel library for LLM serving that uses JIT/AOT compilation for GPU kernels:
- Python APIs in `flashinfer/` delegate to CUDA/C++ kernels in `csrc/` compiled on-demand
- Most kernels are **Jinja templates** (`csrc/*.jinja`) instantiated by `aot_build_utils/generate_*.py` scripts
- Two-stage compile: **JIT** (runtime) and **AOT** (ahead-of-time for wheels)
- Core pattern: **plan/run split** — `plan()` precomputes schedules, `run()` executes with reusable workspaces

### Key Components
```
flashinfer/           # Python API layer
├── attention.py      # BatchAttentionWrapper - plan/run pattern
├── decode.py         # BatchDecodeWithPagedKVCacheWrapper
├── prefill.py        # BatchPrefillWithPagedKVCacheWrapper
├── jit/              # JIT compilation factories (gen_* builders)
├── comm/             # NVSHMEM distributed ops
└── triton/           # Triton kernel alternatives

csrc/                 # CUDA kernel implementations
├── *.jinja           # Templated kernels
├── fused_moe/        # CUTLASS fused MOE (SM80/SM90+)
└── nv_internal/      # TensorRT-LLM integration

aot_build_utils/      # Code generation scripts
└── generate_*.py     # Emit .inc configs and headers
```

## Developer Workflows

### Setup & Build
```bash
# Editable install (JIT mode - compiles on demand)
python -m pip install --no-build-isolation -e . -v

# Set GPU architectures for JIT compilation
export TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0a 10.0a"

# Build AOT kernels (for wheel distribution)
python -m flashinfer.aot --out-dir aot-ops --fa2-head-dim 128,128 --fa3-head-dim 128,128
python -m build --no-isolation --wheel

# Build CMake benchmarks
mkdir build && cp cmake/config.cmake build && cd build && cmake .. && make -j
./bench_single_prefill --help
```

### Testing
```bash
# Run focused tests (GPU required)
pytest -q tests/test_single_prefill.py
pytest -k decode  # Filter by keyword

# Enable torch.compile warmup (torch>=2.4)
FLASHINFER_TEST_TORCH_COMPILE=1 pytest tests/

# CUDA OOMs are auto-skipped - red tests indicate logic bugs
```

### Debugging
```bash
# Clear JIT/AOT caches after kernel changes
python -c "import flashinfer.jit as jit_env; jit_env.clear_cache_dir()"

# Profiler (intra-kernel timeline visualization)
pip install protobuf git+https://github.com/flashinfer-ai/tg4perfetto.git
python profiler/mla.py --batch-size 64 --seq-len 1024
# View *.perfetto-trace at ui.perfetto.dev
```

### Cache Management
- **Location**: `~/.cache/flashinfer/<arch>/` (e.g., `75_80_89_90/`)
- **Override**: Set `FLASHINFER_WORKSPACE_BASE` for containers/CI
- **Structure**: `cached_ops/` (JIT modules), `generated/` (sources)

## Critical Coding Patterns

### 1. Plan/Run Split (Attention Wrappers)
**Why**: Amortize scheduling overhead across batches with variable-length inputs
```python
# flashinfer/decode.py, attention.py, prefill.py
wrapper.plan(...)  # Compute schedule metadata once
wrapper.run(...)   # Reuse workspace, execute kernel
```
**Implementation**: See `BatchDecodeWithPagedKVCacheWrapper` for threading KV indices, LSE tensors, causal flags.

### 2. JIT Kernel Workflow
**Step-by-step to add/modify kernels**:
1. **Template**: Add/edit Jinja in `csrc/*.jinja` OR add generator in `aot_build_utils/`
2. **Factory**: Create `gen_*` JitSpec builder in `flashinfer/jit/` (use `functools.cache` + `build_and_load()`)
3. **Test**: Clear cache + run tests
   ```bash
   python -c "import flashinfer.jit as j; j.clear_cache_dir()" && pytest -q tests/test_single_prefill.py
   ```

**Example**: `flashinfer/jit/activation.py::gen_act_and_mul_module()` shows JitSpec creation pattern.

### 3. Op Registration (torch.library Compatibility)
```python
# flashinfer/utils.py helpers for custom ops
@register_custom_op("flashinfer::my_op", mutates_args=())
def my_op_impl(...): ...

@register_fake_op("flashinfer::my_op")
def my_op_fake(...): ...
```
**Why**: Enables doc builds and older torch versions without torch.library overhead.

### 4. Architecture-Specific Features
- **Backend selection**: `flashinfer/utils.py::determine_attention_backend()`, `is_fa3_backend_supported()`
- **SM80 (Ampere)**: CUTLASS grouped GEMM, no TMA, finalize always separate
- **SM90+ (Hopper/Blackwell)**: TMA support, FP8/FP4 quantization, fused finalize
- **Flags**: Mirror `sm100a_nvcc_flags` when extending Hopper/Blackwell (see `fused_moe.py`, `fp4_quantization.py`)

### 5. NVSHMEM (Distributed)
```python
# flashinfer/comm/nvshmem.py
from flashinfer.comm import nvshmem
nvshmem.init()
# ... collective ops
torch.cuda.synchronize()
nvshmem.finalize()
```
**Setup**: Requires `nvidia-nvshmem-cu12` or `NVSHMEM_INCLUDE_PATH`/`NVSHMEM_LIBRARY_PATH`.

## Tooling & Code Quality

### Linting & Formatting
```bash
./format.sh  # Auto-installs yapf 0.40.2, ruff 0.6.5, codespell 2.3.0, clang-format 15.0.7
```
**Ruff exemptions** (`pyproject.toml`): E402, F405, F403, E741, E501, SIM118, B019, E902, SIM102, SIM108, E731, B020, SIM103

### Build System Details
- **custom_backend.py**: PEP 517 backend that symlinks `3rdparty/`, `csrc/`, `include/` into `flashinfer/data/` for packaging
- **Wheel modes**: 
  - Editable: symlinks to source
  - AOT wheel: packages pre-compiled `.so` from `aot-ops/`

## File Navigation Quickstart

**Starting points**:
- `flashinfer/attention.py` - plan/run pattern & workspace reuse
- `csrc/batch_decode_*.jinja` - kernel template example
- `aot_build_utils/generate_batch_paged_prefill_inst.py` - code generation
- `flashinfer/jit/env.py` - cache paths & NVSHMEM helpers
- `tests/conftest.py` - pytest GPU setup & torch.compile warmup

**Investigating issues**:
- Backend selection → `flashinfer/utils.py`
- Kernel instantiation → `aot_build_utils/generate_*.py`
- JIT factories → `flashinfer/jit/*.py`
- Test patterns → `tests/test_single_prefill.py`, `tests/test_triton_cascade.py`

## Common Pitfalls

1. **Cache staleness**: Always clear cache after modifying `.jinja` templates or NVCC flags
2. **TORCH_CUDA_ARCH_LIST**: Must match target GPU or kernels won't load
3. **OOM vs logic errors**: OOMs are skipped in tests - focus on non-OOM failures
4. **Plan must precede run**: Wrappers require `plan()` before `run()` for workspace allocation
5. **Comment style**: Prefer concise rationale over restating code (see `flashinfer/attention.py` buffer setups)

## Performance & Profiling

- **Intra-kernel profiler**: `profiler/` with perfetto trace generation (experimental, intrusive instrumentation)
- **Benchmark suite**: CMake + nvbench in `benchmarks/` (`bench_batch_decode`, etc.)
- **Key metrics**: SM80 shared mem 96KB, ~64 Tensor Cores/SM; GEMM+finalize <25ms typical

## Integration Points

**Adoption**: vLLM, SGLang, TensorRT-LLM, MLC-LLM, Hugging Face TGI
**Bindings**: PyTorch (primary), TVM, C++ header-only

---

**For detailed SM80 CUTLASS MOE analysis**: See repo docs `DOCUMENT_INDEX.md`, `SM80_CUTLASS_FUSED_MOE_EXECUTION_FLOW.md`

```# FlashInfer Copilot Guide
## Architecture
- flashinfer/__init__.py re-exports high-level APIs backed by CUDA/C++ kernels compiled on demand via flashinfer/jit.* gen_* factories.
- csrc/ holds CUDA implementations; most kernels are parameterized through *.jinja templates with aot_build_utils/generate_*.py emitting instantiation headers and config .inc files.
- BatchAttention and related wrappers (flashinfer/attention.py, decode.py, prefill.py) follow a plan/run split where plan precomputes schedule metadata and run shares preallocated float/int workspaces.
- Dynamic modules land under ~/.cache/flashinfer/* (see flashinfer/jit/env.py); set FLASHINFER_WORKSPACE_BASE to relocate cache when working inside containers.
- Backend/dtype gating lives in flashinfer/utils.py (determine_attention_backend, is_fa3_backend_supported); extend support there alongside kernel changes.
- NVSHMEM/TRT-LLM/vLLM integrations reside in flashinfer/comm/, expecting libnvshmem_host.so and optional NVSHMEM_INCLUDE_PATH or NVSHMEM_LIBRARY_PATH when generating the bindings.
## Build & Packaging
- Editable install: python -m pip install --no-build-isolation -e . -v; custom_backend.py symlinks 3rdparty/, csrc/, include/, and aot-ops into flashinfer/data for packaging.
- CMake/nvbench benchmarks: mkdir build && cp cmake/config.cmake build && cd build && cmake .. && make -j before invoking ./bench_single_prefill --help etc.
- JIT builds honor TORCH_CUDA_ARCH_LIST; set it (e.g. "8.0 8.9 9.0a") before compiling to ensure emitted fatbins match target GPUs.
- Ahead-of-time pipeline: python -m flashinfer.aot --out-dir aot-ops --fa2-head-dim 128,128 --fa3-head-dim 128,128 copies cached .so files into build/aot-ops-package-dir for wheel builds.
- NVSHMEM modules call jit_env.get_nvshmem_include_dirs()/get_nvshmem_lib_dirs(); ensure nvidia-nvshmem is installed or provide paths so gen_nvshmem_module can link -lnvshmem_device.
## Testing & Debug
- GPU-enabled pytest is the norm; run pytest -q tests/test_single_prefill.py or narrow with -k decode to fit available memory.
- Set FLASHINFER_TEST_TORCH_COMPILE=1 to wrap core kernels with torch.compile warmups (requires torch>=2.4); tests/conftest.py caches compiled callables in TORCH_COMPILE_FNS.
- CUDA OOMs are downgraded to pytest.skip, so red tests typically mean logic or numerics rather than capacity.
- Flush stale kernels with flashinfer.jit.clear_cache_dir() whenever changing generated sources or NVCC flags.
- Profiler workflows (profiler/README.md) rely on pip install git+https://github.com/flashinfer-ai/tg4perfetto.git before running python profiler/mla.py --batch-size 64 --seq-len 1024.
## Coding Patterns
- Follow existing plan(...) signature when adding wrappers; BatchDecodeWithPagedKVCacheWrapper (flashinfer/decode.py) shows how to thread kv indices, lse tensors, and causal flags through run().
- New CUDA kernels should expose a gen_* JitSpec builder under flashinfer/jit/* and cache build_and_load() using functools.cache similar to get_holistic_attention_module.
- Register exported ops with flashinfer.utils.register_custom_op/register_fake_op (no-op wrappers that avoid torch.library overhead) so doc builds and older Torch releases continue to import.
- Quantization and fused MOE support (flashinfer/fp4_quantization.py, fused_moe.py) pass sm100a_nvcc_flags and extra include paths; mirror those flags when extending Hopper/Blackwell features.
- Triton kernels live under flashinfer/triton/ with matching module names (activation.py, cascade.py); keep API parity with CUDA variants for test coverage like tests/test_triton_cascade.py.
- Distributed collectives allocate via flashinfer/comm/nvshmem.py; initialize with alloc_empty_unique_id()/init() and call finalize() after torch.cuda.synchronize() just like existing tests.
## Tooling
- format.sh enforces yapf 0.40.2, ruff 0.6.5, codespell 2.3.0, clang-format 15.0.7; it skips auto-yapf but runs codespell --toml pyproject.toml, so expect the script to pip install matching versions.
- Ruff config (pyproject.toml [tool.ruff.lint]) enables E,F,B,SIM checks while ignoring E402,F405,F403,E741,E501,SIM118,B019,E902,SIM102,SIM108,E731,B020,SIM103—match these exemptions when fixing lint.
- Keep new comments purposeful; prefer short rationale above complex buffer setups like in flashinfer/attention.py rather than restating assignments.
