# SharpInfer: A Pure C# LLM Inference Engine Built From Scratch

*Running local AI models in .NET — no Python, no bindings, no compromises.*

---

If you've ever tried to embed a local language model into a .NET application, you know the pain. Every serious inference runtime — llama.cpp, vLLM, Ollama — is written in Python or C++. The .NET ecosystem's options are essentially wrappers: thin managed layers over unmanaged runtimes, with all the friction that implies: native library paths, platform-specific build steps, P/Invoke marshalling, and a debugging experience that disappears the moment execution crosses the language boundary.

**SharpInfer is a different answer to that problem.** It's a full LLM inference engine written entirely in C#, from the binary file parser to the attention kernels to the REST API. No Python. No llama.cpp bindings. No ONNX Runtime. Just managed .NET code running transformer inference end to end.

---

## Why Build This in C#?

The obvious question is: why not just call llama.cpp from C#? The honest answer is that wrapping a C++ library solves the immediate problem while creating a dozen new ones.

When your inference code is managed, your entire stack is debuggable in Visual Studio. You can set breakpoints inside the attention mechanism. You can profile memory allocation in the dequantizer. You can write unit tests for individual transformer components. You can ship a single `dotnet publish` artifact that runs on Linux, Windows, and macOS without a native library matrix.

There's also the integration story. A pure C# engine embeds into ASP.NET APIs, worker services, desktop apps, and Azure Functions as naturally as any other NuGet dependency. No sidecar processes, no IPC, no container orchestration required just to call a model.

The tradeoff is real — a well-tuned C++ kernel will beat managed code on raw throughput for large batch workloads. But for the enormous class of .NET applications that need *embedded* local inference — an IDE assistant, a document processor, a customer service agent — the integration simplicity is worth more than the last 20% of throughput.

---

## What It Actually Does

SharpInfer implements the full transformer inference pipeline from scratch:

**GGUF and Safetensors loading.** The engine parses GGUF binary files (v2 and v3) and Safetensors files directly, including sharded multi-file models. Metadata, tokenizer vocabulary, and tensor descriptors are all extracted from the file format without any external tooling.

**Dequantization.** GGUF models ship in quantized formats to reduce file size and memory footprint. SharpInfer implements the full GGML quantization spec in managed code — every format from the legacy `Q4_0` through the K-quant family (`Q4_K_M`, `Q6_K`, etc.) that most modern model releases use, plus FP16 and BF16. The packed scale decoding for K-quants in particular required careful bit-level reconstruction from the GGML spec.

**The transformer forward pass.** RMS normalization, token embeddings, grouped-query attention (GQA) with rotary position embeddings (RoPE), gated MLP layers (SiLU and GELU activations), and the output projection — all implemented in C#. The KV cache stores key and value projections across generation steps so autoregressive decoding doesn't recompute the full context on every token.

**Architecture flexibility.** The loader handles the naming and layout differences between model families. Phi-3 uses fused QKV weights stored under different tensor names than LLaMA. Falcon-40B has a second attention norm for parallel attention. Gemma 2 adds post-attention and post-FFN layer norms. SharpInfer detects and adapts to these differences at load time rather than requiring separate codepaths for each architecture.

**Sampling.** The generation pipeline supports temperature scaling, top-K filtering, nucleus (top-P) sampling, repetition penalty, stop sequences, and seeded reproducible generation.

---

## The API

SharpInfer ships with an ASP.NET Core REST API that is wire-compatible with the OpenAI Chat Completions API. That compatibility matters in practice: it means Continue.dev, Open WebUI, and any client library built for OpenAI work against SharpInfer with a single URL change.

```bash
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Explain RoPE embeddings."}],
    "temperature": 0.7,
    "stream": true
  }'
```

Streaming uses Server-Sent Events, exactly as the OpenAI spec defines. The Python `openai` library, LangChain, and LlamaIndex all connect without modification.

Beyond the OpenAI-compatible endpoints, the API exposes model management: pull models from HuggingFace, load them into memory, inspect metadata, and clean up orphaned files. The `/api/pull` endpoint accepts `hf.co/` prefixed repo names with quant filters, so you can pull exactly the model variant you want:

```bash
curl -X POST http://localhost:3512/api/pull \
  -d '{"name": "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"}'
```

---

## Modelfile: Declarative Model Packaging

SharpInfer introduces a `Modelfile` format — a Dockerfile-inspired declarative config that bundles a model with its system prompt, sampling parameters, chat template, and LoRA adapters into a single reproducible definition:

```dockerfile
FROM ./models/llama-3.2-8b.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER max_tokens 1024

SYSTEM "You are a helpful coding assistant specializing in C# and .NET."

ADAPTER ./adapters/code-lora.bin
```

Modelfiles can be bundled into `.simodel` archives — zip files containing the Modelfile and all referenced assets — for distribution and version control.

---

## Beyond the Basics

The feature set goes well past basic inference. Several capabilities that are difficult to compose on top of library wrappers are first-class in SharpInfer:

**Speculative decoding** runs a small draft model ahead of the main model to generate token candidates, then verifies them in parallel. In practice this yields 2–3x generation speedup for models where a good draft model exists.

**LoRA adapter hot-swapping** loads fine-tuned PEFT adapters and applies them to the base model at inference time. Adapters can be swapped between requests, which makes it practical to run a single base model with multiple specialized personalities.

**Prompt caching** serializes the KV state for a prompt prefix and reloads it on subsequent requests. For applications with a fixed long system prompt — a detailed persona, a large codebase context, a lengthy document — this eliminates the cost of re-encoding that prefix on every generation.

**Multi-agent orchestration** is built into the core library, supporting chain, parallel, debate, router, handoff, and map-reduce agent topologies. Each node in the graph is a model call; the orchestration layer handles routing, aggregation, and state passing.

**Tool calling and MCP** implement the OpenAI function-calling spec and the emerging Model Context Protocol for external tool integration.

---

## Hardware Backends

CPU inference is the default and works everywhere .NET runs. For GPU acceleration, SharpInfer includes a CUDA backend that offloads matrix operations to NVIDIA GPUs via P/Invoke to compiled native kernels. Metal (Apple Silicon), Vulkan (cross-platform GPU), and ARM NEON backends are also included, with automatic detection and selection at startup.

The GPU backend is opt-in and the engine is fully functional without it — which matters for server deployments where you may not control the hardware.

---

## Getting Started

The quickest way to run SharpInfer is with Docker:

```bash
git clone https://github.com/your-org/sharpinfer
cd sharpinfer
docker compose up
```

The API is then live on port 3512. Pull a model and start generating:

```bash
# Pull a quantized LLaMA 3.1 8B
curl -X POST http://localhost:3512/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"}'

# Load it
curl -X POST http://localhost:3512/api/load \
  -H "Content-Type: application/json" \
  -d '{"name": "Meta-Llama-3.1-8B-Instruct"}'

# Chat
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"m","messages":[{"role":"user","content":"Hello from C#!"}]}'
```

For direct embedding in a .NET project, reference `SharpInfer.Core` and wire up the `InferenceEngine` class directly — no API server required.

---

## The Bigger Picture

SharpInfer sits at an intersection that hasn't had good options until now: local LLM inference that is genuinely first-class in the .NET ecosystem, not bolted on from the outside.

The codebase is intentionally readable. The GGUF parser, the attention implementation, the dequantization routines — they're all straightforward C# that you can step through, extend, and modify. If you want to experiment with a custom attention variant, implement a new sampling strategy, or support a model architecture that doesn't fit the standard LLaMA mold, you're working with real managed code rather than fighting FFI boundaries.

For .NET developers who've been watching the local AI wave from the sidelines because the tooling assumed Python, SharpInfer makes the full inference pipeline a native citizen of the platform you already know.

---

*SharpInfer is MIT-licensed. The full source, documentation, and Docker setup are in the repository.*
