---
layout: default
title: Getting Started
nav_order: 2
has_children: true
permalink: /getting-started/
---

# Getting Started
{: .no_toc }

Get SharpInfer running in under five minutes.
{: .fs-6 .fw-300 }

---

## Prerequisites

| Requirement | Details |
|---|---|
| Operating System | Windows 10/11, macOS 13+, or Linux |
| .NET Runtime | .NET 8.0 SDK or later |
| RAM | 8 GB minimum (16 GB recommended for 7B+ models) |
| Disk Space | 2–30+ GB per model, depending on size and quantization |
| GPU *(optional)* | NVIDIA with CUDA, Apple Silicon (Metal), or any Vulkan 1.2+ GPU |

---

## Installation

### 1. Install .NET 8

Download and install the .NET 8 SDK from [dotnet.microsoft.com/download](https://dotnet.microsoft.com/download/dotnet/8.0).

Verify the installation:

```bash
dotnet --version
# 8.0.xxx
```

### 2. Get SharpInfer

Clone the repository or download the latest release:

```bash
git clone https://github.com/jpaulduncan/SharpInfer
cd SharpInfer
dotnet build
```

### 3. Get a Model

The fastest way is the built-in model puller. Start the API, then pull a model:

```bash
# Start the API server
dotnet run --project src/SharpInfer.Api -- --port 3512 --models-dir ./models

# In another terminal — pull a ~4 GB quantized Llama 3.1 8B
curl -X POST http://localhost:3512/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"}'
```

Or download any `.gguf` file manually from [HuggingFace](https://huggingface.co/models?library=gguf) and place it in `./models/`.

{: .note }
**First model?** Try **Phi-3 Mini** (≈2 GB) — it loads fast and produces high-quality output for its size.
`hf.co/bartowski/Phi-3-mini-4k-instruct-GGUF:Q4_K_M`

---

## Two Ways to Run

SharpInfer can run as an **interactive CLI** for direct conversation, or as an **API server** for integration with other tools.

### CLI — Interactive Chat

```bash
dotnet run --project src/SharpInfer.Cli -- --model ./models/your-model.gguf
```

You'll see a `You >` prompt. Type and press Enter. The response streams back token by token.

→ See the full [CLI Guide]({% link cli/index.md %}) for all options and commands.

### API Server

```bash
dotnet run --project src/SharpInfer.Api -- --port 3512 --models-dir ./models
```

The server starts with no model loaded. Use `/api/pull` to download a model, then `/api/load` to activate it. Once a model is loaded, send chat requests to `/v1/chat/completions`.

→ See the [API Reference]({% link api/index.md %}) for full endpoint documentation.

### Docker

```bash
docker compose up        # CPU
docker compose --profile gpu up    # NVIDIA GPU
```

The API is available on port 3512. The `./models` directory is mounted as a volume so downloaded models persist.

---

## Your First Chat

Once a model is loaded, try this:

```bash
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sharpinfer",
    "messages": [{"role": "user", "content": "Hello! What can you help me with?"}],
    "max_tokens": 256
  }'
```

You should receive a JSON response with the model's reply in `choices[0].message.content`.

{: .highlight }
The API is fully OpenAI-compatible. Any tool that connects to OpenAI — Continue.dev, Open WebUI, LangChain, the `openai` Python library — works against SharpInfer with `base_url = "http://localhost:3512/v1"`.
