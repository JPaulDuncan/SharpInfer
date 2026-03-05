---
layout: home
title: Home
nav_order: 1
---

# SharpInfer
{: .fs-9 }

A pure C# LLM inference engine — no Python, no llama.cpp bindings, no ONNX Runtime.
{: .fs-6 .fw-300 }

[Get Started]({% link getting-started/index.md %}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/jpaulduncan/SharpInfer){: .btn .fs-5 .mb-4 .mb-md-0 }

---

SharpInfer loads GGUF and Safetensors models directly, dequantizes weights in managed code, and runs the full transformer forward pass natively on .NET 8. Because the entire stack is managed C#, it embeds cleanly into any .NET application, debugs in Visual Studio, and deploys anywhere .NET runs.

The API is OpenAI-compatible, so tools like Continue.dev, Open WebUI, and any OpenAI client library work out of the box.

---

## Feature Highlights

### Model Formats
SharpInfer loads **GGUF** (v2/v3), **Safetensors** (including sharded multi-file models), **GPTQ**, and **AWQ** without any external dependencies. Tokenizer vocabulary and merge rules are extracted from the model file automatically.

### Full Quantization Support
Every GGML quantization format is supported: the K-quant family (`Q2_K` through `Q6_K`), legacy formats (`Q4_0`, `Q5_0`, `Q8_0`), and standard precision types (`F32`, `F16`, `BF16`).

### OpenAI-Compatible API
The REST API mirrors the OpenAI Chat Completions spec. Streaming uses Server-Sent Events. Any client library, tool, or script that targets OpenAI works against SharpInfer with a URL change.

### Hardware Backends
CPU inference works everywhere. Optional GPU acceleration via **CUDA** (NVIDIA), **Metal** (Apple Silicon), **Vulkan** (cross-platform), and **ARM NEON** — with automatic backend detection.

### Advanced Capabilities

| Capability | Description |
|---|---|
| Speculative decoding | 2–3× speedup using a draft model |
| LoRA adapters | Hot-swap fine-tuned adapters at runtime |
| Prompt caching | Persist KV state across requests |
| RAG | Retrieval-augmented generation with pluggable vector stores |
| Multi-agent orchestration | Chain, parallel, debate, router, handoff, and map-reduce flows |
| Tool calling | OpenAI function-calling spec + MCP client |
| Structured output | JSON schema enforcement |
| Multimodal vision | CLIP ViT-L/14 and SigLIP image encoders |

---

## Quick Navigation

<div class="grid-of-cards">

**[Getting Started]({% link getting-started/index.md %})**
Install .NET, pull a model, and have your first conversation in under five minutes.

**[CLI Guide]({% link cli/index.md %})**
All command-line flags, interactive slash commands, and configuration file reference.

**[API Reference]({% link api/index.md %})**
Complete REST API documentation — endpoints, request/response formats, streaming, WebSocket, and batch processing.

**[Integration Guide]({% link integration/index.md %})**
Connect SharpInfer to Continue.dev, Open WebUI, Python, JavaScript, and custom frontends.

**[VS Code Extension]({% link vscode/index.md %})**
Local AI code completion and chat inside Visual Studio Code via the JSON-RPC language server.

**[Modelfile]({% link modelfile.md %})**
Declarative model packaging — bundle a model with its system prompt, parameters, and adapters.

</div>

---

## Why C#?

Most local inference tools are Python wrappers around C++ libraries. SharpInfer takes a different approach: the entire pipeline lives in managed .NET, which means:

- **Full debuggability** — step through attention and dequantization in Visual Studio
- **Native .NET embedding** — reference `SharpInfer.Core` like any NuGet package
- **Single-binary deployment** — `dotnet publish` produces a self-contained executable
- **Cross-platform without a native library matrix** — works on Linux, Windows, and macOS out of the box

The tradeoff is real — a hand-tuned C++ kernel beats managed code on raw throughput. But for applications that need embedded local inference inside a .NET service, the integration simplicity outweighs the performance gap for most use cases.
