---
layout: default
title: Command-Line Options
parent: CLI Guide
nav_order: 1
---

# Command-Line Options
{: .no_toc }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Basic Options

| Flag | Description |
|---|---|
| `--model <path>` | Path to the model file (GGUF, Safetensors, Modelfile, or .simodel). Required unless using `--config`. |
| `--config <path>` | Load all settings from a JSON configuration file. |
| `--generate-config <path>` | Write a fully-annotated default config to a file, then exit. |
| `--help`, `-h` | Show the help message. |
| `--context <n>` | Override the context window size (tokens). Default is model-dependent. |
| `--models-dir <path>` | Directory for downloaded models. Defaults to `~/.sharpinfer/models`. |
| `--hf-token <token>` | HuggingFace API token for downloading gated/private models. |

---

## Generation Parameters

| Flag | Default | Description |
|---|---|---|
| `--temp <value>` | `0.7` | Sampling temperature. `0.0` = deterministic, `1.0+` = more creative. |
| `--top-p <value>` | `0.9` | Nucleus sampling threshold. Tokens outside the top cumulative probability are excluded. |
| `--top-k <value>` | `40` | Limits selection to the top K most likely tokens at each step. |
| `--max-tokens <n>` | `512` | Maximum tokens to generate per response. |
| `--repeat-penalty <v>` | `1.1` | Penalizes recently used tokens. `1.0` = no penalty. |
| `--seed <n>` | random | Fixed random seed for reproducible generation. |
| `--system <prompt>` | — | System prompt. Sets the model's persona and behavior. |

---

## GPU and Hardware

| Flag | Description |
|---|---|
| `--gpu` | Enable NVIDIA CUDA GPU acceleration. |
| `--gpu-device <id>` | Select a GPU by device ID (default: `0`). For multi-GPU systems. |
| `--gpu-layers <n>` | Number of transformer layers to offload to GPU. `-1` = all layers. |
| `--backend <name>` | Force a specific backend: `auto`, `cpu`, `cuda`, `metal`, `vulkan`, `arm_neon`. |
| `--flash-attention` | Enable FlashAttention for faster, more memory-efficient inference. |

---

## Advanced Feature Flags

| Flag | Description |
|---|---|
| `--speculative` | Enable speculative decoding for faster generation. Requires `--draft-model`. |
| `--draft-model <path>` | Path to the smaller draft model used for speculative decoding. |
| `--lora <name:path>` | Load a LoRA adapter. Format: `name:path/to/adapter.bin`. Repeat for multiple adapters. |
| `--prompt-cache` | Enable prompt caching (persists KV state to disk for fast repeated prompts). |
| `--rag` | Enable retrieval-augmented generation. |
| `--rag-docs <path>` | Directory or file path to ingest into the RAG knowledge base on startup. |
| `--tools` | Enable tool use (web search, URL reader). |
| `--beam-search <n>` | Enable beam search with N beams for higher quality, slower output. |
| `--structured-output` | Enable structured output mode (JSON schema enforcement). |
| `--multimodal` | Enable vision / multimodal capabilities for image understanding. |
| `--batch-processing` | Enable the batch processing endpoint for bulk offline inference. |

---

## Enterprise Flags

| Flag | Description |
|---|---|
| `--auth` | Enable API key authentication. All requests must include `Authorization: Bearer <key>`. |
| `--rate-limit` | Enable per-key rate limiting with token bucket algorithm. |
| `--audit` | Enable JSONL audit logging for every request. |
| `--metering` | Enable usage metering with per-key daily token tracking. |

---

## Examples

```bash
# Interactive chat with GPU
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/llama-3.1-8b.gguf \
  --gpu --temp 0.7 --max-tokens 1024

# Deterministic output with a seed
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/mistral-7b.gguf \
  --temp 0.0 --seed 42

# With a coding system prompt
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/codellama-7b.gguf \
  --system "You are an expert C# developer. Be concise."

# Speculative decoding for faster generation
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/llama-3.1-8b.gguf \
  --speculative --draft-model ./models/llama-3.2-1b.gguf

# Load a LoRA adapter
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/base-model.gguf \
  --lora "code-assist:./adapters/code-lora.bin"
```
