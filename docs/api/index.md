---
layout: default
title: API Reference
nav_order: 4
has_children: true
permalink: /api/
---

# API Reference
{: .no_toc }

OpenAI-compatible REST API for local LLM inference.
{: .fs-6 .fw-300 }

---

## Overview

The SharpInfer API server exposes two groups of endpoints:

1. **OpenAI-compatible** — `/v1/chat/completions`, `/v1/models`. Wire-compatible with any OpenAI client library.
2. **Model management** — `/api/pull`, `/api/load`, `/api/tags`, `/api/delete`, and more. Download, activate, and manage models at runtime.

An interactive **Swagger UI** is available at:

```
http://localhost:3512/swagger
```

---

## Server Startup

```bash
# Basic (no model preloaded)
dotnet run --project src/SharpInfer.Api -- --port 3512 --models-dir ./models

# Preload a model at startup
dotnet run --project src/SharpInfer.Api -- --model ./models/llama-3.1-8b.gguf --port 3512

# With GPU
dotnet run --project src/SharpInfer.Api -- --model ./models/llama-3.1-8b.gguf --gpu --port 3512
```

Or with Docker:

```bash
docker compose up
```

---

## Connection Details

| Property | Value |
|---|---|
| Default Port | `3512` |
| Base URL | `http://localhost:3512` |
| OpenAI-compatible Base URL | `http://localhost:3512/v1` |
| Authentication | None by default (enterprise auth is opt-in) |
| CORS | Fully permissive — all origins allowed |

---

## All Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server health and loaded model info |
| `GET` | `/v1/models` | List available models (OpenAI format) |
| `POST` | `/v1/chat/completions` | Chat completions — streaming and non-streaming |
| `GET` | `/api/tags` | List downloaded model files |
| `POST` | `/api/pull` | Download a model from HuggingFace (NDJSON stream) |
| `POST` | `/api/load` | Load a downloaded model into memory |
| `DELETE` | `/api/delete` | Delete a downloaded model file |
| `POST` | `/api/show` | Show model metadata and parameters |
| `GET` | `/api/orphans` | List model files not referenced by any Modelfile |
| `DELETE` | `/api/orphans` | Delete orphaned model files |

---

## In This Section

- **[Endpoints]({% link api/endpoints.md %})** — Complete reference for every endpoint
- **[Streaming]({% link api/streaming.md %})** — Server-Sent Events and WebSocket streaming
- **[Batch Processing]({% link api/batch.md %})** — Submit bulk jobs and retrieve results asynchronously
- **[Enterprise Middleware]({% link api/enterprise.md %})** — Auth, rate limiting, audit logging, and usage metering
