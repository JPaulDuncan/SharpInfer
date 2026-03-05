# SharpInfer

A pure C# LLM inference engine built from scratch — no Python, no llama.cpp bindings, no ONNX Runtime. SharpInfer loads GGUF and Safetensors models directly, dequantizes weights in managed code, and runs the full transformer forward pass natively on .NET 8.

## Why SharpInfer?

Most local inference tools are Python wrappers around C++ libraries. SharpInfer takes a different approach: the entire inference pipeline — tokenization, attention, sampling, and generation — is implemented in C# from the ground up. This makes it straightforward to embed in .NET applications, extend with custom logic, and deploy anywhere .NET runs.

The API is OpenAI-compatible, so tools like Continue.dev, Open WebUI, and any OpenAI client library work out of the box.

## Features

**Core Inference**
- Full transformer forward pass in managed C# (embedding, RoPE, grouped-query attention, SiLU/GELU MLP, RMS normalization)
- Streaming and non-streaming text generation
- KV cache for efficient autoregressive decoding
- FlashAttention-style block-wise computation

**Model Format Support**
- **GGUF** (v2/v3) — with embedded tokenizer extraction
- **Safetensors** — including sharded multi-file models
- **GPTQ** — INT4/INT8 group-wise quantization (auto-detected from safetensors)
- **AWQ** — activation-aware INT4 quantization (auto-detected from safetensors)
- **Modelfile** — declarative model packaging format (similar to a Dockerfile)
- **.simodel** — bundled zip archive with Modelfile and all referenced files

**Quantization**
- K-quant family: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
- Legacy GGML: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
- Standard: F32, F16, BF16, FP8 (e4m3fn)
- GPTQ and AWQ dequantization from safetensors

**Sampling Pipeline**
- Temperature scaling
- Top-K filtering
- Top-P (nucleus) sampling
- Repetition penalty
- Stop sequences (token IDs and strings)
- Reproducible generation via seed

**Advanced Capabilities**
- Speculative decoding (2–3x speedup with draft model)
- LoRA adapter loading and hot-swapping (HuggingFace PEFT format)
- Prompt caching (persistent KV state serialization with GZip compression)
- Retrieval-Augmented Generation (pluggable vector store backends)
- Classifier-Free Guidance
- Beam search with length/repetition penalties
- Multimodal vision (CLIP ViT-L/14, SigLIP encoders)
- Tool calling with JSON schema
- MCP (Model Context Protocol) client
- Multi-agent orchestration (chain, parallel, debate, router, handoff, map-reduce)
- Structured output (JSON schema enforcement)

**Hardware Backends**
- CPU (default, all platforms)
- CUDA (NVIDIA GPUs via P/Invoke to native kernels)
- Metal (Apple Silicon)
- Vulkan (cross-platform GPU)
- ARM NEON (ARM processors)
- Automatic backend detection and selection

## Project Structure

```
SharpInfer/
├── src/
│   ├── SharpInfer.Core/          Core inference engine
│   │   ├── Agents/               Multi-agent orchestration
│   │   ├── Engine/               InferenceEngine, ModelPuller, Modelfile, PromptCache
│   │   ├── Layers/               Transformer, FlashAttention, RotaryEmbedding
│   │   ├── Mcp/                  Model Context Protocol client
│   │   ├── Models/               GGUF/Safetensors loaders, ModelConfig
│   │   ├── Multimodal/           Vision encoders (CLIP, SigLIP)
│   │   ├── Sampling/             SamplingPipeline, GenerationConfig
│   │   ├── Tensors/              Tensor ops, quantization, compute backends
│   │   ├── Tokenizer/            BPE tokenizer (HuggingFace + SentencePiece)
│   │   └── Tools/                Tool registry and execution
│   ├── SharpInfer.Gpu/           CUDA backend (P/Invoke + native kernels)
│   ├── SharpInfer.Api/           REST API server (ASP.NET Core)
│   ├── SharpInfer.Cli/           Interactive chat CLI
│   └── SharpInfer.VsCode/        VS Code language server
├── Documents/                    Detailed guides (API, CLI, Integration, VS Code)
├── Dockerfile                    Multi-stage build for the API
├── docker-compose.yml            CPU and GPU service definitions
└── LICENSE.txt                   MIT License
```

## Quick Start

### Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- A GGUF or Safetensors model file

### Run the CLI

```bash
# Interactive chat with a local model
dotnet run --project src/SharpInfer.Cli -- --model ./models/my-model.gguf

# With GPU acceleration
dotnet run --project src/SharpInfer.Cli -- --model ./models/my-model.gguf --gpu

# Generate a default config file, then customize it
dotnet run --project src/SharpInfer.Cli -- --generate-config sharpinfer.json
dotnet run --project src/SharpInfer.Cli -- --config sharpinfer.json
```

### Run the API Server

```bash
# Start the API (no model pre-loaded — use /api/pull and /api/load to manage models)
dotnet run --project src/SharpInfer.Api -- --port 3512 --models-dir ./models

# Pre-load a model on startup
dotnet run --project src/SharpInfer.Api -- --model ./models/my-model.gguf --port 3512

# With GPU
dotnet run --project src/SharpInfer.Api -- --model ./models/my-model.gguf --gpu --port 3512
```

### Run with Docker

```bash
# Build and start (CPU)
docker compose up

# Pull and load a model via the API
curl -X POST http://localhost:3512/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"}'

curl -X POST http://localhost:3512/api/load \
  -H "Content-Type: application/json" \
  -d '{"name": "Meta-Llama-3.1-8B-Instruct"}'

# Start with GPU (requires NVIDIA Container Toolkit)
docker compose --profile gpu up
```

Or build the image directly:

```bash
docker build -t sharpinfer .
docker run -p 3512:3512 -v ./models:/models sharpinfer
```

## API Reference

The API server exposes OpenAI-compatible endpoints and model management endpoints. Swagger UI is available at `http://localhost:3512/swagger` when the server is running.

### Chat Completions (OpenAI-compatible)

```bash
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain transformers in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": true
  }'
```

Streaming responses use Server-Sent Events (SSE), matching the OpenAI format.

### All Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health status, loaded model info, models directory |
| `GET` | `/v1/models` | List available models (OpenAI format) |
| `POST` | `/v1/chat/completions` | Chat completions (streaming and non-streaming) |
| `GET` | `/api/tags` | List all downloaded models with size, format, digest |
| `POST` | `/api/pull` | Download a model from HuggingFace (NDJSON progress) |
| `DELETE` | `/api/delete` | Delete a downloaded model |
| `POST` | `/api/show` | Show model metadata and engine parameters |
| `POST` | `/api/load` | Load a model into the inference engine |
| `GET` | `/api/orphans` | List orphaned models (not referenced by any Modelfile) |
| `DELETE` | `/api/orphans` | Clean up orphaned models (dry-run by default) |

### Model Management

Models can be pulled from HuggingFace using several name formats:

```bash
# HuggingFace repo with quant filter
curl -X POST http://localhost:3512/api/pull \
  -d '{"name": "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"}'

# Built-in alias
curl -X POST http://localhost:3512/api/pull -d '{"name": "llama3"}'

# Direct HuggingFace repo (auto-selects best GGUF)
curl -X POST http://localhost:3512/api/pull \
  -d '{"name": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"}'
```

Built-in aliases include `llama3`, `llama2`, `mistral`, `mixtral`, `codellama`, `phi2`, `gemma`, `qwen2`, and `tinyllama`.

## Modelfile

SharpInfer supports a declarative model packaging format inspired by Dockerfile syntax. A Modelfile bundles a model with its configuration, system prompt, chat template, and adapters into a reproducible setup.

```dockerfile
FROM ./models/llama-3.2-8b.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER max_tokens 1024

SYSTEM "You are a helpful coding assistant specializing in C# and .NET."

TEMPLATE "[INST] {{.System}}\n{{.Prompt}} [/INST]"

ADAPTER ./adapters/code-lora.bin

LICENSE MIT
```

Modelfiles can be bundled into `.simodel` archives (zip format) for distribution.

## Configuration

The CLI supports a comprehensive JSON configuration file that controls all engine features. Generate a documented default with:

```bash
dotnet run --project src/SharpInfer.Cli -- --generate-config sharpinfer.json
```

Key configuration sections:

| Section | Controls |
|---------|----------|
| `model` | Model path, context length, format |
| `generation` | Temperature, top_p, top_k, max_tokens, repetition penalty, seed |
| `gpu` | Enable/disable, device ID, number of GPU layers |
| `speculative` | Draft model path, lookahead tokens |
| `lora` | Adapter paths, active adapter selection |
| `promptCache` | Enable/disable, cache directory |
| `rag` | Document paths, chunk size, vector store backend |
| `tools` | Web search API key, URL reader |
| `agents` | Multi-agent flow definitions |
| `hardware` | Backend selection (auto/cuda/metal/vulkan/neon/cpu) |
| `quantization` | Dynamic requantization settings |
| `multimodal` | Vision model path, image settings |

## CLI Options

```
Usage: sharpinfer --model <path> [options]
       sharpinfer --config <path>
       sharpinfer --generate-config <path>

Model Loading:
  --model, -m <path>      Path to model file (GGUF, Safetensors, Modelfile, .simodel)
  --config <path>         Load settings from JSON config file

Generation:
  --temperature <float>   Sampling temperature (default: 0.7)
  --top-p <float>         Nucleus sampling threshold (default: 0.9)
  --top-k <int>           Top-K token filtering
  --max-tokens <int>      Maximum tokens to generate (default: 512)
  --repeat-penalty <f>    Repetition penalty (default: 1.1)

GPU:
  --gpu                   Enable CUDA GPU acceleration
  --gpu-layers <int>      Number of layers to offload to GPU

Advanced:
  --context, -c <int>     Override context length
  --models-dir <path>     Models directory for management commands
  --hf-token <token>      HuggingFace API token for gated models
```

## GPU Acceleration

SharpInfer includes a CUDA backend that offloads matrix multiplication, softmax, normalization, and activation functions to NVIDIA GPUs.

### Building the CUDA Kernels

```bash
cd src/SharpInfer.Gpu/Kernels
nvcc -shared -o sharpinfer_cuda.so kernels.cu -O3
```

On Windows, compile to `sharpinfer_cuda.dll` instead. Place the compiled library where .NET can find it (alongside the application DLL or in a system library path).

### Docker GPU Setup

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

```bash
docker compose --profile gpu up
```

The GPU service runs on port 8080 by default and automatically enables `--gpu`.

## Integrations

SharpInfer's OpenAI-compatible API works with a wide range of tools.

### Continue.dev (VS Code AI Coding)

In `.continue/config.json`:

```json
{
  "models": [{
    "title": "SharpInfer",
    "provider": "openai",
    "model": "sharpinfer",
    "apiBase": "http://localhost:3512/v1"
  }]
}
```

### Open WebUI

```bash
# Add SharpInfer as a connection in Open WebUI settings
# URL: http://localhost:3512/v1
# No API key required
```

### Python (openai library)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3512/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="sharpinfer",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### curl

```bash
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"m","messages":[{"role":"user","content":"Hi"}]}'
```

## Architecture

SharpInfer implements the transformer architecture from first principles in C#:

1. **Tokenization** — BPE tokenizer supporting both HuggingFace (merge-rank) and SentencePiece (score-based) formats. Vocabularies and merge rules are extracted from model files automatically.

2. **Embedding** — Token IDs are mapped to dense vectors via the embedding weight matrix.

3. **Transformer Blocks** — Each layer applies RMS normalization, multi-head attention with rotary position embeddings (RoPE), and a gated MLP (SiLU activation). Grouped-query attention (GQA) is supported for models with fewer KV heads than query heads.

4. **KV Cache** — Key and value projections are cached across generation steps to avoid redundant computation during autoregressive decoding.

5. **Sampling** — Logits from the output projection are processed through a configurable pipeline: repetition penalty, temperature scaling, top-K filtering, top-P nucleus sampling, then categorical sampling.

6. **Dequantization** — Quantized weights are dequantized on-the-fly during weight loading. The K-quant routines (Q4_K, Q6_K, etc.) implement the full GGML block format spec including packed scale decoding.

## Documentation

Detailed guides are available in the `Documents/` directory:

| Guide | Contents |
|-------|----------|
| `SharpInfer_API_Guide.md` | Full REST API reference, streaming, batch processing, enterprise features |
| `SharpInfer_CLI_Guide.md` | CLI usage, all command-line options, interactive commands, troubleshooting |
| `SharpInfer_Integration_Reference.md` | Frontend integration, endpoint details, code examples |
| `SharpInfer_VSCode_Extension_Guide.md` | VS Code language server setup and usage |

## License

MIT — see [LICENSE.txt](LICENSE.txt).
