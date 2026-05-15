# llama.cpp-triattention-mtp

## 中文

`llama.cpp` fork，集成 **Qwen3 native MTP speculative decoding** 和 **TriAttention**，用于长上下文推理实验。

建议仓库描述：

```text
llama.cpp fork with Qwen3 native MTP (PR #22673) + TriAttention for long-context inference
```

建议 Topics：

```text
llama.cpp, qwen3, mtp, triattention, cuda
```

## 特性

- Qwen3 原生 MTP：`--spec-type draft-mtp --spec-draft-n-max 2`
- TriAttention 长上下文推理路径
- Windows CUDA 构建已验证：MSVC + CUDA 13.0
- 128K context 已在 RTX 4090 Laptop 16 GB + 96 GB RAM 上跑通
- 预编译 Windows 二进制通过 GitHub Releases 分发

## 编译

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

## 运行

```bash
llama-server \
  -m Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 37 -c 131072
```

Windows 启动模板见 [scripts/llama-server-start.bat](scripts/llama-server-start.bat)。

## 实测

测试环境：RTX 4090 Laptop GPU 16 GB、i9-13980HX、96 GB RAM、Windows 11、CUDA 13.0、Qwen3.6-27B Q4_K_M。

| 配置 | 速度 | 说明 |
| --- | ---: | --- |
| `-ngl 37 -c 131072`，无 MTP | ~6.6 t/s | 27B Q4_K_M 接近 16 GB 显存上限 |
| `-ngl 37 -c 131072`，`draft-mtp` | ~3.6 t/s | MTP 接受率高，但额外 draft context 让显存饱和 |
| `-ngl 30 -c 8192`，`draft-mtp` | ~2.9 t/s | 降低 GPU offload 后 CPU 占比上升 |

MTP 接受率约 **78-91%**。这说明 draft、验证、接受路径已经跑通；16 GB 显存环境适合做功能验证，不适合展示 MTP 最佳性能。要验证速度收益，建议使用 24 GB+ GPU 或更小模型。

详细数据见 [docs/benchmarks.md](docs/benchmarks.md)。

## Releases

Windows 预编译二进制见 [GitHub Releases](https://github.com/desmondreedhf-afk/llama.cpp-triattention-mtp/releases)。

- `llama-server-triattention-mtp.exe`
- `scripts/llama-server-start.bat`

## License

MIT License，沿用 upstream [llama.cpp](https://github.com/ggerganov/llama.cpp) 的授权和署名。

---

## English TL;DR

`llama.cpp` fork with Qwen3 native MTP speculative decoding (PR #22673) and TriAttention for long-context inference experiments.

Highlights:

- Qwen3 native MTP: `--spec-type draft-mtp --spec-draft-n-max 2`
- TriAttention long-context inference path
- Windows CUDA build tested with MSVC + CUDA 13.0
- 128K context tested on RTX 4090 Laptop 16 GB + 96 GB RAM
- Windows pre-built binaries are published through GitHub Releases

Build:

```bash
git clone --recurse-submodules https://github.com/desmondreedhf-afk/llama.cpp-triattention-mtp.git
cd llama.cpp-triattention-mtp
mkdir build && cd build
cmake .. -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_ARCHITECTURES=89
cmake --build . --config Release --target llama-server --parallel
```

Run:

```bash
llama-server -m Qwen3.6-27B-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 2 -ngl 37 -c 131072
```

Observed MTP acceptance is about **78-91%** on Qwen3.6-27B. The 16 GB RTX 4090 Laptop test system validates functionality, but a 24 GB+ GPU or smaller model is recommended for performance testing.
