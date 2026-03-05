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

- **[Command-Line Options]({% link cli/options.md %})** — All `--flags` grouped by category
- **[Interactive Commands]({% link cli/commands.md %})** — Slash commands you type during a session
- **[Configuration File]({% link cli/configuration.md %})** — Full JSON config reference
- **[Model Management]({% link cli/model-management.md %})** — Downloading and switching models
- **[Generation Parameters]({% link cli/generation-parameters.md %})** — Temperature, top-p, repetition penalty explained
- **[GPU Acceleration]({% link cli/gpu.md %})** — CUDA, Metal, Vulkan, and ARM NEON backends
- **[Advanced Features]({% link cli/advanced.md %})** — Speculative decoding, LoRA, RAG, agents, and more
