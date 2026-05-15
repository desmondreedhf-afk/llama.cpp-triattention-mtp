# llama.cpp-triattention-mtp

## 中文说明

这是一个面向长上下文推理实验的 `llama.cpp` 分支，合并了两条主要能力：

- **TriAttention**：面向长上下文的注意力优化与 KV cache 管理实验。
- **Qwen3 原生 MTP speculative decoding**：合并自 upstream PR #22673，可通过 `draft-mtp` 使用 Qwen3 系列模型自带的 multi-token prediction 头。

这个仓库的目标不是发布一个通用的一键整合包，而是保留一个可复现、可编译、便于继续实验的源码工作区。预编译二进制会放在 GitHub Releases 中，源码仓库本身不提交本地编译产物或模型权重。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| Windows CUDA 编译 | 已验证，MSVC + CUDA 13.0 |
| Linux CUDA 编译 | 保留 upstream 构建路径 |
| TriAttention | 已集成 |
| Qwen3 MTP (`draft-mtp`) | 已集成并跑通 |
| 128K 上下文 | 在 RTX 4090 Laptop 16 GB + 96 GB RAM 上验证过 |
| 预编译二进制 | 通过 Releases 分发 |

## 功能亮点

- TriAttention CUDA 路径位于 `ggml/src/ggml-cuda/`
- Qwen3 native MTP speculative decoding，启动参数：
  `--spec-type draft-mtp --spec-draft-n-max 2`
- TurboQuant KV cache 类型：`turbo2_0`、`turbo3_0`、`turbo4_0`
- `ggml_turbo_wht` graph op，用于 Walsh-Hadamard Transform 相关路径
- 保持接近 upstream `llama.cpp` 的目录结构，便于继续合并和对照

## 从源码编译

```bash
git clone --recurse-submodules https://github.com/desmondreedhf-afk/llama.cpp-triattention-mtp.git
cd llama.cpp-triattention-mtp
mkdir build && cd build
```

Windows (MSVC + CUDA):

```bash
cmake .. -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_ARCHITECTURES=89
cmake --build . --config Release --target llama-server --parallel
```

Linux:

```bash
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-server --parallel -j$(nproc)
```

## 运行示例

```bash
llama-server \
  -m Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 37 -c 131072
```

Windows 用户可以参考 [scripts/llama-server-start.bat](scripts/llama-server-start.bat)，把 `MODEL_PATH` 改成自己的模型路径后运行。

## 实测结论

测试环境：RTX 4090 Laptop GPU 16 GB、i9-13980HX、96 GB RAM、Windows 11、CUDA 13.0、Qwen3.6-27B Q4_K_M。

| 配置 | 速度 | 说明 |
| --- | ---: | --- |
| `-ngl 37 -c 131072`，无 MTP | ~6.6 t/s | 27B Q4_K_M 接近 16 GB 显存上限 |
| `-ngl 37 -c 131072`，`draft-mtp` | ~3.6 t/s | MTP 接受率高，但额外 draft context 让显存饱和 |
| `-ngl 30 -c 8192`，`draft-mtp` | ~2.9 t/s | 降低 GPU offload 后 CPU 占比上升 |

MTP 接受率实测约 **78-91%**，说明 draft 路径、验证路径和接受逻辑已经跑通。当前 16 GB 显存环境不适合用来展示 MTP 的最佳性能；它更适合证明功能可用。要验证性能收益，建议使用 24 GB 或更大显存的 GPU，或换用更小模型。

详细数据见 [docs/benchmarks.md](docs/benchmarks.md)。

## Releases

Windows 预编译二进制会放在 [GitHub Releases](https://github.com/desmondreedhf-afk/llama.cpp-triattention-mtp/releases)。

- `llama-server-triattention-mtp.exe`：带 TriAttention + MTP 的 server 二进制
- `scripts/llama-server-start.bat`：启动模板

源码仓库不会提交 `.exe`、模型权重、构建目录或本机路径配置。

## 许可证

本仓库沿用 upstream `llama.cpp` 的 MIT License，并保留 upstream 作者署名与相关文档。

---

## English

This repository is a `llama.cpp` fork for long-context inference experiments. It combines two main pieces of work:

- **TriAttention**: attention and KV-cache management experiments for long-context inference.
- **Qwen3 native MTP speculative decoding**: merged from upstream PR #22673 and exposed through `draft-mtp`, using the native multi-token prediction heads in compatible Qwen3 models.

The repository is intended to stay reproducible and source-first. Local build outputs and model weights are not committed. Pre-built binaries are distributed through GitHub Releases.

## Status

| Item | Status |
| --- | --- |
| Windows CUDA build | Tested with MSVC + CUDA 13.0 |
| Linux CUDA build | Upstream build path preserved |
| TriAttention | Integrated |
| Qwen3 MTP (`draft-mtp`) | Integrated and functionally validated |
| 128K context | Tested on RTX 4090 Laptop 16 GB + 96 GB RAM |
| Pre-built binaries | Distributed through Releases |

## Highlights

- TriAttention CUDA path under `ggml/src/ggml-cuda/`
- Qwen3 native MTP speculative decoding:
  `--spec-type draft-mtp --spec-draft-n-max 2`
- TurboQuant KV cache types: `turbo2_0`, `turbo3_0`, `turbo4_0`
- `ggml_turbo_wht` graph op for Walsh-Hadamard Transform paths
- Directory layout remains close to upstream `llama.cpp` for easier comparison and future merges

## Build From Source

```bash
git clone --recurse-submodules https://github.com/desmondreedhf-afk/llama.cpp-triattention-mtp.git
cd llama.cpp-triattention-mtp
mkdir build && cd build
```

Windows (MSVC + CUDA):

```bash
cmake .. -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_ARCHITECTURES=89
cmake --build . --config Release --target llama-server --parallel
```

Linux:

```bash
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-server --parallel -j$(nproc)
```

## Run Example

```bash
llama-server \
  -m Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 37 -c 131072
```

Windows users can use [scripts/llama-server-start.bat](scripts/llama-server-start.bat) as a launch template.

## Benchmark Summary

Test system: RTX 4090 Laptop GPU 16 GB, i9-13980HX, 96 GB RAM, Windows 11, CUDA 13.0, Qwen3.6-27B Q4_K_M.

| Config | Speed | Notes |
| --- | ---: | --- |
| `-ngl 37 -c 131072`, no MTP | ~6.6 t/s | 27B Q4_K_M is already close to the 16 GB VRAM limit |
| `-ngl 37 -c 131072`, `draft-mtp` | ~3.6 t/s | High MTP acceptance, but the extra draft context saturates VRAM |
| `-ngl 30 -c 8192`, `draft-mtp` | ~2.9 t/s | Lower GPU offload increases CPU-bound work |

Observed MTP acceptance is about **78-91%**, which confirms that the draft, verification, and acceptance paths are working. The 16 GB test machine is useful for functional validation, not for best-case MTP performance claims. For speedup validation, use a 24 GB+ GPU or a smaller model.

See [docs/benchmarks.md](docs/benchmarks.md) for details.

## Releases

Windows pre-built binaries are available on the [GitHub Releases](https://github.com/desmondreedhf-afk/llama.cpp-triattention-mtp/releases) page.

- `llama-server-triattention-mtp.exe`: server binary with TriAttention + MTP
- `scripts/llama-server-start.bat`: launch template

The repository does not commit `.exe` files, model weights, build directories, or local machine paths.

## License

This repository follows upstream `llama.cpp` under the MIT License and preserves upstream author attribution and documentation.
