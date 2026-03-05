---
layout: default
title: Configuration File
parent: CLI Guide
nav_order: 3
---

# Configuration File
{: .no_toc }

Store all settings in a JSON file instead of passing flags every time.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Generating a Config

```bash
dotnet run --project src/SharpInfer.Cli -- --generate-config sharpinfer.json
```

This creates `sharpinfer.json` with every available option documented in-line. Open it in any editor.

Use it with:

```bash
dotnet run --project src/SharpInfer.Cli -- --config sharpinfer.json
```

---

## Config Sections

### `model`

Controls which model is loaded.

```json
"model": {
  "path": "./models/llama-3.1-8b-Q4_K_M.gguf",
  "contextLength": 4096,
  "format": "auto"
}
```

### `generation`

Controls how text is generated.

```json
"generation": {
  "temperature": 0.7,
  "topP": 0.9,
  "topK": 40,
  "maxTokens": 512,
  "repetitionPenalty": 1.1,
  "seed": null,
  "systemPrompt": "You are a helpful assistant."
}
```

### `gpu`

Hardware acceleration settings.

```json
"gpu": {
  "enabled": true,
  "deviceId": 0,
  "layers": -1,
  "flashAttention": true
}
```

Set `layers` to `-1` to offload all layers to GPU, or a positive integer to offload only that many (useful when GPU VRAM is limited).

### `hardware`

Force a specific compute backend or let SharpInfer choose.

```json
"hardware": {
  "backend": "auto"
}
```

Options: `auto`, `cpu`, `cuda`, `metal`, `vulkan`, `arm_neon`.

### `speculative`

Speculative decoding for 2–3× faster generation.

```json
"speculative": {
  "enabled": true,
  "draftModelPath": "./models/llama-3.2-1b.gguf",
  "lookaheadTokens": 5
}
```

### `lora`

LoRA adapter configuration.

```json
"lora": {
  "adapters": [
    { "name": "code-assist", "path": "./adapters/code-lora.bin" },
    { "name": "formal-tone", "path": "./adapters/formal-lora.bin" }
  ],
  "active": "code-assist"
}
```

Use `/lora apply <name>` at runtime to switch adapters.

### `promptCache`

Persist the KV state for a system prompt so repeated sessions start instantly.

```json
"promptCache": {
  "enabled": true,
  "cacheDir": "./.cache/prompt"
}
```

### `rag`

Retrieval-augmented generation.

```json
"rag": {
  "enabled": true,
  "documentPaths": ["./docs/", "./knowledge-base/manual.pdf"],
  "backend": "in-memory",
  "topK": 5,
  "chunkSize": 512,
  "chunkOverlap": 64,
  "similarityThreshold": 0.7
}
```

### `tools`

Web search and URL reader tools.

```json
"tools": {
  "enabled": true,
  "webSearchApiKey": "YOUR_SEARCH_API_KEY",
  "urlReader": true
}
```

### `mcpServers`

Model Context Protocol server connections.

```json
"mcpServers": [
  {
    "name": "filesystem",
    "url": "http://localhost:3001"
  }
]
```

### `multiAgent`

Multi-agent orchestration flows.

```json
"multiAgent": {
  "agents": [
    {
      "name": "researcher",
      "role": "You are a research specialist.",
      "modelPath": "./models/llama-3.1-8b.gguf"
    },
    {
      "name": "writer",
      "role": "You are a professional technical writer.",
      "modelPath": "./models/mistral-7b.gguf"
    }
  ],
  "flows": [
    {
      "name": "research-and-write",
      "type": "chain",
      "steps": ["researcher", "writer"]
    }
  ]
}
```

Flow types: `chain`, `parallel`, `debate`, `router`, `handoff`, `review-loop`.

### `enterprise`

Authentication, rate limiting, logging, and metering.

```json
"enterprise": {
  "auth": { "enabled": false },
  "rateLimit": { "enabled": false, "tokensPerSecond": 10, "burstCapacity": 50 },
  "audit": { "enabled": false, "logPath": "./logs/audit.jsonl" },
  "metering": { "enabled": false }
}
```

### `multimodal`

Vision/image capabilities.

```json
"multimodal": {
  "enabled": false,
  "visionModelPath": "./models/clip-vit-l14.bin",
  "encoder": "clip"
}
```

---

## Example: Complete Config

```json
{
  "model": {
    "path": "./models/llama-3.1-8b-Q4_K_M.gguf",
    "contextLength": 8192
  },
  "generation": {
    "temperature": 0.7,
    "topP": 0.9,
    "maxTokens": 1024,
    "systemPrompt": "You are a helpful assistant."
  },
  "gpu": {
    "enabled": true,
    "layers": -1,
    "flashAttention": true
  },
  "speculative": {
    "enabled": true,
    "draftModelPath": "./models/llama-3.2-1b.gguf"
  },
  "promptCache": {
    "enabled": true,
    "cacheDir": "./.cache"
  }
}
```
