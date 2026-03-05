---
layout: default
title: Quick Start — Docker
parent: Getting Started
nav_order: 1
---

# Quick Start — Docker
{: .no_toc }

The fastest path from zero to a working local LLM.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Step 1: Start the Server

```bash
git clone https://github.com/jpaulduncan/SharpInfer
cd SharpInfer
docker compose up
```

The API starts immediately on port **3512** with no model loaded.

---

## Step 2: Pull a Model

Download a quantized model from HuggingFace. This streams download progress:

```bash
curl -X POST http://localhost:3512/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "hf.co/bartowski/Phi-3-mini-4k-instruct-GGUF:Q4_K_M"}'
```

You'll see progress output as bytes download. The model file lands in `./models/`.

{: .note }
**Model size reference:**
- Phi-3 Mini Q4_K_M — ~2.2 GB — great for quick tests
- Llama 3.1 8B Q4_K_M — ~4.9 GB — excellent general-purpose model
- Mistral 7B Q4_K_M — ~4.4 GB — fast, strong reasoning

---

## Step 3: Load the Model

```bash
curl -X POST http://localhost:3512/api/load \
  -H "Content-Type: application/json" \
  -d '{"name": "Phi-3-mini-4k-instruct"}'
```

Response:
```json
{"status": "success", "model": "Phi-3-mini-4k-instruct-Q4_K_M.gguf"}
```

The model is now in memory and ready.

---

## Step 4: Chat

```bash
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sharpinfer",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain what a transformer model is in two sentences."}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

You'll receive a JSON response with the model's reply.

---

## Step 5 *(optional)*: Enable Streaming

Add `"stream": true` to see tokens appear one by one:

```bash
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sharpinfer",
    "messages": [{"role": "user", "content": "Count to ten slowly."}],
    "stream": true
  }'
```

Tokens arrive as `data: {...}` Server-Sent Events, ending with `data: [DONE]`.

---

## Step 6 *(optional)*: Connect Open WebUI

For a full chat UI:

```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:3512/v1 \
  -e OPENAI_API_KEY=none \
  --name open-webui ghcr.io/open-webui/open-webui:main
```

Open [http://localhost:3000](http://localhost:3000) — SharpInfer appears as your local model provider.

---

## GPU Acceleration

With an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed:

```bash
docker compose --profile gpu up
```

The GPU service starts on port **8080** with CUDA acceleration enabled. Expect 3–10× faster generation depending on your GPU.

---

## What's Next

- [CLI Guide]({% link cli/index.md %}) — Run interactive chat directly from the terminal
- [API Reference]({% link api/index.md %}) — Full endpoint documentation
- [Integration Guide]({% link integration/index.md %}) — Connect to Continue.dev, Open WebUI, Python, JavaScript
