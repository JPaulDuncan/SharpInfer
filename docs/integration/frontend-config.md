---
layout: default
title: Frontend Configuration
parent: Integration Guide
nav_order: 3
---

# Frontend Configuration
{: .no_toc }

Everything you need to connect a custom chat frontend to SharpInfer.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Connection Settings

| Setting | Value |
|---|---|
| **API Base URL** | `http://localhost:3512/v1` |
| **API Key** | Any non-empty string — e.g., `"none"` or `"sk-local"`. Not validated unless enterprise auth is enabled. |
| **Model name** | Any string — e.g., `"sharpinfer"`. The server uses whatever model is currently loaded. |
| **CORS** | Fully permissive — all origins, methods, and headers allowed. |

---

## Configuration Checklist

When connecting a frontend to SharpInfer, configure these settings:

- [ ] **API Base URL:** `http://localhost:3512/v1`
- [ ] **API Key:** any non-empty value
- [ ] **Model name:** any string (`"sharpinfer"` is conventional)
- [ ] **Streaming:** supported — enable SSE streaming for real-time token-by-token display
- [ ] **System prompt:** send as the first message with `"role": "system"`
- [ ] **Conversation history:** the server is **stateless** — your frontend must send the full message history on every request
- [ ] **Stop sequences:** pass as the `stop` array field
- [ ] **Token usage:** returned in `usage` field of non-streaming responses

---

## Stateless Server

SharpInfer resets its context after every request — it has no memory of previous conversations. Your frontend is responsible for maintaining history and sending it with each request:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a mutex?"},
    {"role": "assistant", "content": "A mutex is a synchronization primitive..."},
    {"role": "user", "content": "How does it differ from a semaphore?"}
  ]
}
```

---

## Differences from OpenAI API

| Aspect | OpenAI | SharpInfer |
|---|---|---|
| Base URL | `https://api.openai.com/v1` | `http://localhost:3512/v1` |
| Authentication | Required (Bearer token) | Optional — off by default |
| Model selection | Chooses from hosted models | Uses the loaded model; name is ignored |
| `n` parameter | Multiple completions | Not supported — always returns 1 choice |
| `functions` / `tools` | Supported | Not supported via REST (available in CLI config) |
| `logprobs` | Supported | Not supported |
| Rate limits | Per-account cloud limits | Optional per-key limiting (enterprise) |
| Context | Managed server-side | Stateless — reset each request |
| Cost | Per-token billing | Free — runs locally |

---

## Model Management Workflow

A typical frontend integration handles the full model lifecycle:

```
1. GET  /health           → Check if server is running
2. GET  /api/tags         → List downloaded models
3. POST /api/pull         → Download a model (stream progress)
4. POST /api/load         → Load model into memory
5. POST /v1/chat/completions → Chat!
6. POST /api/load         → Hot-swap to a different model
7. GET  /api/orphans      → Find unused models
8. DELETE /api/orphans    → Clean up disk space
```

---

## Error Handling

| HTTP Status | Meaning | Recommended action |
|---|---|---|
| `200` | Success | Parse response normally |
| `400` | Bad request | Check JSON format; ensure `messages` is a non-empty array |
| `404` | Model not found | Check model name; use `GET /api/tags` to see available models |
| `503` | No model loaded | Call `POST /api/load` before sending chat requests |
| `500` | Inference error | Retry after a delay; check server logs if persistent |

Error responses use this format:
```json
{"error": "Human-readable error message"}
```

---

## Docker Network Note

When running SharpInfer in Docker and your frontend in another container, use the Docker hostname instead of `localhost`:

```
http://sharpinfer:3512/v1          # If using docker compose service name
http://host.docker.internal:3512/v1  # If frontend is also in Docker
```

For host-to-container access, `http://localhost:3512/v1` works normally.
