# FlashInfer Copilot Guide
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
