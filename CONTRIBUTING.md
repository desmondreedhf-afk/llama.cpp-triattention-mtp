# Contributing

This repository is currently being prepared for a focused `llama.cpp` research branch. Contribution guidance will become more specific after the source tree is imported.

## Before Opening a Change

- Keep experimental changes clearly separated from upstream synchronization work.
- Include hardware, compiler, and model details for benchmark-related changes.
- Avoid committing model weights, generated binaries, or large benchmark artifacts.
- Preserve upstream license and attribution notices when importing or modifying `llama.cpp` files.

## Useful PR Context

A good pull request should explain:

- what changed
- why the change is needed
- how it was tested
- whether it affects performance, compatibility, or model output

## Benchmark Notes

For performance claims, include enough detail for another person to reproduce the result:

- CPU/GPU model
- operating system
- compiler and build flags
- model name and quantization
- prompt/decode settings
- baseline commit or branch
