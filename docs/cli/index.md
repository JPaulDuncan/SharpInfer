---
layout: default
title: CLI Guide
nav_order: 3
has_children: true
permalink: /cli/
---

# CLI Guide
{: .no_toc }

Run interactive LLM conversations directly from your terminal.
{: .fs-6 .fw-300 }

---

## Overview

The SharpInfer CLI launches an interactive chat session against any supported model file. All engine features — GPU acceleration, LoRA adapters, RAG, speculative decoding, multi-agent flows — are available through command-line flags or a JSON configuration file.

### Launch

```bash
dotnet run --project src/SharpInfer.Cli -- --model path/to/model.gguf
```

You'll see a `You >` prompt. Type your message and press Enter. The response streams back token by token.

### Using a Config File

For complex setups, generate a documented config and customize it:

```bash
# Generate a fully-annotated config with all defaults
dotnet run --project src/SharpInfer.Cli -- --generate-config sharpinfer.json

# Launch using that config
dotnet run --project src/SharpInfer.Cli -- --config sharpinfer.json
```

---

## In This Section

- **[Command-Line Options](options)** — All `--flags` grouped by category
- **[Interactive Commands](commands)** — Slash commands you type during a session
- **[Configuration File](configuration)** — Full JSON config reference
- **[Model Management](model-management)** — Downloading and switching models
- **[Generation Parameters](generation-parameters)** — Temperature, top-p, repetition penalty explained
- **[GPU Acceleration](gpu)** — CUDA, Metal, Vulkan, and ARM NEON backends
- **[Advanced Features](advanced)** — Speculative decoding, LoRA, RAG, agents, and more
