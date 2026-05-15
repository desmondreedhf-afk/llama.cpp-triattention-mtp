# Design Notes

## TriAttention

- Custom CUDA attention kernels in `ggml/src/ggml-cuda/`
- Optimized for long-context sequences with reduced VRAM footprint
- Compatible with Flash Attention API (`-fa on`)
- See `docs/TRIATTENTION.md` and `docs/TRIATTENTION-API.md` for details

## MTP (Multi-Token Prediction)

- Merged from upstream llama.cpp PR #22673 (`am17an/mtp-clean`)
- Activates via `--spec-type draft-mtp --spec-draft-n-max 2`
- Creates a draft context using Qwen3's native MTP heads, not a separate draft model
- Acceptance rates: 78-91% on Qwen3.6-27B
- VRAM overhead: additional KV cache and compute buffers for the draft context

## Merge Architecture

```text
upstream llama.cpp
  -> TriAttention fork (iamwavecut/llama-cpp-triattention)
      -> PR #22673 MTP merge (this repo)
```

Minimal conflicts during merge: 48 files auto-merged, with two manual fixes:

- `ggml_turbo_wht` graph op implementation re-added after merge
- `#include <stdbool.h>` guarded for MSVC C++ builds

## TurboQuant

- KV cache compression via PolarQuant + QJL (arXiv 2504.19874)
- Types: `turbo2_0` (2-bit), `turbo3_0` (3-bit), `turbo4_0` (4-bit)
- Used as `--cache-type-k turboN_0` in llama-server
- CUDA kernels in `ggml/src/ggml-cuda/turbo-innerq.cu`

## Known Limitations

- 16 GB VRAM is not sufficient for 27B MTP acceleration; 24 GB+ is recommended
- Windows MSVC: `stdbool.h` workaround added (`#ifndef __cplusplus` guard in `ggml.h`)
- TriAttention templates require CMake regeneration after code changes
- TurboQuant WHT op has a graph-level stub; full CPU fallback is not yet implemented

## Upstream Compatibility

- Tracks upstream llama.cpp master as of the merge commit
- TriAttention and MTP changes are additive; no core API changes
- Re-merging with newer upstream may require re-applying build fixes
