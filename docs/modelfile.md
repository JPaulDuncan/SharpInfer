---
layout: default
title: Modelfile
nav_order: 7
---

# Modelfile
{: .no_toc }

Declarative model packaging — bundle a model with its system prompt, parameters, and adapters.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Overview

A **Modelfile** is a plain text configuration file, similar in spirit to a `Dockerfile`, that describes a complete model setup. It references a base model and layers configuration on top: sampling parameters, a system prompt, a chat template, LoRA adapters, and a license.

Modelfiles make model setups reproducible and shareable. Bundle a Modelfile with its referenced assets into a `.simodel` archive for distribution.

---

## Syntax

```dockerfile
FROM <model-source>

PARAMETER <name> <value>
...

SYSTEM "<system prompt text>"

TEMPLATE "<chat template>"

ADAPTER <path-to-adapter>

LICENSE <license-name-or-text>
```

---

## Directives

### `FROM`

Specifies the base model. The `FROM` line is required.

```dockerfile
# Local file
FROM ./models/llama-3.1-8b-Q4_K_M.gguf

# HuggingFace repo (downloaded automatically if not present)
FROM hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M

# Built-in alias
FROM llama3

# Another Modelfile (inherit and extend)
FROM ./base-assistant.modelfile
```

---

### `PARAMETER`

Sets a generation parameter. Multiple `PARAMETER` lines are allowed.

| Parameter | Type | Description |
|---|---|---|
| `temperature` | float | Sampling temperature. Default: `0.7` |
| `top_p` | float | Nucleus sampling threshold. Default: `0.9` |
| `top_k` | integer | Top-K filter. Default: `40` |
| `max_tokens` | integer | Max tokens to generate. Default: `512` |
| `repetition_penalty` | float | Repetition penalty. Default: `1.1` |
| `seed` | integer | Fixed seed for reproducible output |
| `stop` | string | Add a stop sequence (repeat for multiple) |
| `context_length` | integer | Override context window size |

```dockerfile
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER max_tokens 1024
PARAMETER repetition_penalty 1.05
PARAMETER stop "<|end|>"
PARAMETER stop "<|endoftext|>"
```

---

### `SYSTEM`

Sets the system prompt — the instruction given to the model at the start of every conversation.

```dockerfile
SYSTEM "You are a helpful assistant specializing in C# and .NET development. Provide concise, idiomatic code examples and explain the reasoning behind design decisions."
```

Multi-line system prompts use a heredoc-style block:

```dockerfile
SYSTEM """
You are a senior software engineer with expertise in:
- C# and .NET 8
- Cloud architecture (Azure, AWS)
- Distributed systems

Provide practical, production-ready advice. Be concise but thorough.
"""
```

---

### `TEMPLATE`

Defines the chat template used to format messages for the model. If omitted, SharpInfer auto-detects the template from GGUF metadata.

```dockerfile
# Phi-3 format
TEMPLATE "<|system|>\n{{.System}}<|end|>\n<|user|>\n{{.Prompt}}<|end|>\n<|assistant|>"

# LLaMA 3 instruct format
TEMPLATE "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{{.System}}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{{.Prompt}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# ChatML format (Mistral, Qwen, etc.)
TEMPLATE "<|im_start|>system\n{{.System}}<|im_end|>\n<|im_start|>user\n{{.Prompt}}<|im_end|>\n<|im_start|>assistant"
```

Template variables:
- `{{.System}}` — the system prompt text
- `{{.Prompt}}` — the user message

---

### `ADAPTER`

Loads a LoRA adapter on top of the base model.

```dockerfile
ADAPTER ./adapters/code-style.bin
ADAPTER ./adapters/formal-tone.bin
```

Multiple `ADAPTER` lines are allowed. All adapters listed are loaded at startup.

---

### `LICENSE`

Documents the model's license. This is informational and doesn't affect behavior.

```dockerfile
LICENSE MIT

# Or a full SPDX identifier
LICENSE Apache-2.0

# Or inline text for custom licenses
LICENSE "This model may only be used for non-commercial research purposes."
```

---

## Complete Examples

### Coding Assistant

```dockerfile
FROM hf.co/bartowski/CodeLlama-7B-Instruct-GGUF:Q4_K_M

PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER max_tokens 2048

SYSTEM """
You are an expert software engineer. When asked to write code:
- Use idiomatic patterns for the language
- Include brief inline comments for non-obvious logic
- Prefer readability over cleverness
- Flag potential bugs or edge cases proactively
"""

LICENSE llama2
```

### Customer Support Bot

```dockerfile
FROM ./models/llama-3.1-8b-Q4_K_M.gguf

PARAMETER temperature 0.4
PARAMETER top_p 0.9
PARAMETER max_tokens 512
PARAMETER repetition_penalty 1.1

SYSTEM "You are a friendly and professional customer support agent for Acme Corp. Answer questions about our products clearly and concisely. If you don't know the answer, say so and offer to escalate to a human agent."

ADAPTER ./adapters/acme-product-knowledge.bin

LICENSE MIT
```

### Creative Writer

```dockerfile
FROM llama3

PARAMETER temperature 0.9
PARAMETER top_p 0.95
PARAMETER top_k 80
PARAMETER max_tokens 2048
PARAMETER repetition_penalty 1.05

SYSTEM "You are a creative writing assistant with a vivid, engaging style. Help users develop stories, characters, and worlds. Ask clarifying questions to understand their vision before writing."

LICENSE MIT
```

---

## .simodel Archives

A `.simodel` file is a zip archive containing a Modelfile and all its referenced assets — model weights, adapters, and any other files.

```
my-assistant.simodel
├── Modelfile
├── model.gguf
└── adapters/
    └── custom-style.bin
```

Load a `.simodel` directly:

```bash
dotnet run --project src/SharpInfer.Cli -- --model ./my-assistant.simodel
```

Export your current setup to a Modelfile at runtime:

```
/export-modelfile ./my-setup.modelfile
```

`.simodel` archives are ideal for sharing fine-tuned setups, deploying to new machines, and version-controlling model configurations.
