---
layout: default
title: Model Management
parent: CLI Guide
nav_order: 4
---

# Model Management
{: .no_toc }

Downloading, listing, and switching models.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Supported Model Formats

| Format | Description |
|---|---|
| **GGUF** | The most common format for local inference. Models from TheBloke, bartowski, and the wider llama.cpp ecosystem use this format. Supports all quantization levels. |
| **Safetensors** | HuggingFace's native format. Safe, fast to load, and widely available. Supports sharded multi-file models. |
| **GPTQ** | Quantized models using group-wise INT4/INT8 compression. Auto-detected from safetensors. |
| **AWQ** | Activation-aware weight quantization — high quality INT4 models. Auto-detected from safetensors. |
| **Modelfile** | A declarative config file that references a base model and adds parameters, system prompts, and adapters. See [Modelfile reference](../modelfile). |
| **.simodel** | A zip archive bundling a Modelfile and all its referenced assets into a single distributable file. |

---

## Downloading Models

### Built-in Aliases

SharpInfer ships with short aliases for popular models. At the `You >` prompt:

```
/pull llama3
/pull mistral
/pull phi3
/pull codellama
```

Type `/pull aliases` to see all available aliases.

### HuggingFace Repos

Download a specific model by HuggingFace repo name:

```
/pull TheBloke/Mistral-7B-Instruct-v0.2-GGUF
```

SharpInfer automatically selects the best GGUF variant for your available RAM.

### Specific Quantization

Use the `hf.co/` prefix with a `:quant` suffix to pick a specific quantization:

```
/pull hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M
/pull hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q8_0
```

| Quant | Size (8B) | Notes |
|---|---|---|
| `Q2_K` | ~3 GB | Smallest. Noticeably lower quality. |
| `Q4_0` | ~4.5 GB | Legacy format. OK quality. |
| `Q4_K_M` | ~4.9 GB | **Recommended default.** Best quality/size trade-off. |
| `Q5_K_M` | ~5.7 GB | Better quality, slightly larger. |
| `Q6_K` | ~6.6 GB | Near-lossless. Good if you have VRAM. |
| `Q8_0` | ~8.5 GB | Excellent quality. Large. |
| `F16` | ~16 GB | Full precision. Maximum quality. Needs significant RAM/VRAM. |

### Gated Models

Some models on HuggingFace require accepting a license agreement. Log in at [huggingface.co](https://huggingface.co), accept the model's terms, then pass your token:

```bash
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/... --hf-token hf_yourtoken123
```

---

## Managing Downloads

| Command | Description |
|---|---|
| `/pull list` | List all downloaded models with sizes and format. |
| `/pull jobs` | Show active and completed download jobs. |
| `/pull delete <name>` | Delete a downloaded model file. |
| `/model list` | Show models available to load. |

---

## Switching Models

To switch models mid-session, use `/pull` to download a new model, then restart with `--model`:

```bash
dotnet run --project src/SharpInfer.Cli -- --model ./models/new-model.gguf
```

The API server supports hot-swapping via `POST /api/load` without restarting.

---

## Where Models Are Stored

By default, models download to `~/.sharpinfer/models/`. Override this with `--models-dir`:

```bash
dotnet run --project src/SharpInfer.Cli -- --models-dir /data/models --model ...
```

In Docker, the `./models` directory is mounted as a volume, so downloaded models persist across container restarts.
