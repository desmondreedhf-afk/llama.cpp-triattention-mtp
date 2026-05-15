# llama.cpp-triattention-mtp

Experimental `llama.cpp` research workspace for TriAttention and multi-token prediction (MTP) exploration.

This repository is being prepared as a focused fork/workbench for inference-side experiments around attention optimization, speculative decoding ideas, and model execution paths related to MTP. The implementation files will be added after the local project is moved into this repository.

## Goals

- Keep changes close to upstream `llama.cpp` structure where possible.
- Prototype TriAttention-related kernels, scheduling, or execution paths in a reproducible way.
- Explore MTP-friendly decoding flows without hiding benchmark assumptions.
- Make experiments easy to compare against a clean upstream baseline.

## Current Status

This repository is in bootstrap state. The project metadata and documentation are being prepared first; source files, build instructions, and benchmark notes will be filled in once the codebase is uploaded.

## Planned Layout

```text
.
|-- README.md
|-- docs/
|   |-- notes.md
|   `-- benchmarks.md
|-- examples/
|-- scripts/
`-- src/
```

The final layout may follow upstream `llama.cpp` more closely after the source tree is added.

## Development Notes

When the code lands, the recommended workflow will be documented here, including:

- supported build targets
- required compiler/CMake versions
- model compatibility notes
- benchmark commands and hardware details
- differences from upstream `llama.cpp`

## Roadmap

- [ ] Import the working source tree.
- [ ] Document build and run commands.
- [ ] Add baseline benchmark instructions.
- [ ] Document TriAttention/MTP design assumptions.
- [ ] Track upstream compatibility notes.

## Upstream

This project is intended to build on ideas and structure from [`llama.cpp`](https://github.com/ggerganov/llama.cpp). Upstream credits, license details, and any fork-specific notices should be preserved when the source tree is imported.
