# Benchmarks

## Environment

| Field | Value |
| --- | --- |
| CPU | 13th Gen Intel Core i9-13980HX, 24 threads |
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU |
| RAM | 96 GB DDR5 |
| VRAM | 16 GB |
| OS | Windows 11 |
| Compiler | MSVC 19.44 + CUDA 13.0 |
| CMake | 4.3 with Ninja |
| Model | Qwen3.6-27B Q4_K_M, about 15.2 GB |
| Backend | llama.cpp triattention-mtp with PR #22673 merged |

## Results: MTP vs Baseline

### Without MTP

```bash
llama-server -m model.gguf -ngl 37 -c 131072
```

| Metric | Value |
| --- | ---: |
| Prompt eval | 5-180 t/s |
| Token generation | ~6.6 t/s |
| Memory (VRAM) | ~15.9 GB / 16 GB |

### With MTP (draft-mtp)

```bash
llama-server -m model.gguf -ngl 37 -c 131072 --spec-type draft-mtp --spec-draft-n-max 2
```

| Metric | Value |
| --- | ---: |
| Prompt eval | 0.19-2.95 t/s, CPU-bound |
| Token generation | 3.6 t/s |
| Draft acceptance | 90.9% (120/132) |
| Memory (VRAM) | ~15.9 GB / 16 GB, saturated |

### Reduced Context (MTP)

```bash
llama-server -m model.gguf -ngl 30 -c 8192 --spec-type draft-mtp --spec-draft-n-max 2
```

| Metric | Value |
| --- | ---: |
| Prompt eval | 5.3 t/s |
| Token generation | 2.9 t/s |
| Draft acceptance | 78.6% (176/224) |

## Conclusion

The MTP code path is working. Draft acceptance rates of 78-91% confirm that the server is producing and accepting draft tokens.

16 GB VRAM is the bottleneck for this specific setup. The `draft-mtp` mode creates an additional MTP draft context on the same model, adding KV cache and compute-buffer overhead. On a 16 GB card already near saturation with a 27B Q4_K_M model, this overhead can push work to CPU and negate MTP's decode savings.

The 16 GB test machine is useful for functional validation, not for measuring best-case MTP speedup. A 24 GB+ GPU should be used for performance claims.
