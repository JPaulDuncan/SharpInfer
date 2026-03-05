---
layout: default
title: Endpoints
parent: API Reference
nav_order: 1
---

# Endpoints
{: .no_toc }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Health Check

```
GET /health
```

Returns server status and information about the currently loaded model. Use this to check if the server is running and whether a model is ready.

**Response — model loaded:**
```json
{
  "status": "ok",
  "model": "LLaMA 3.1 8B Q4_K_M (8192 ctx)"
}
```

**Response — no model loaded:**
```json
{
  "status": "no_model",
  "message": "No model loaded. Use POST /api/pull then POST /api/load."
}
```

---

## List Models

```
GET /v1/models
```

Returns available models in OpenAI format. Tools like Continue.dev and Open WebUI call this endpoint to discover your model.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3.1-8b-Q4_K_M",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

If no model is loaded, `data` is an empty array.

---

## Chat Completions

```
POST /v1/chat/completions
Content-Type: application/json
```

The main inference endpoint. Accepts a conversation history and returns the model's response. Compatible with the OpenAI Chat Completions API.

### Request Fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | Yes | — | Any string — the loaded model is used regardless. |
| `messages` | array | Yes | — | Array of `{role, content}` objects. Roles: `system`, `user`, `assistant`. |
| `temperature` | float | No | `0.7` | Sampling temperature. Range: `0.0–2.0`. |
| `top_p` | float | No | `0.9` | Nucleus sampling threshold. Range: `0.0–1.0`. |
| `top_k` | integer | No | `40` | Top-K token filter. |
| `max_tokens` | integer | No | `512` | Maximum tokens to generate. |
| `stream` | boolean | No | `false` | If `true`, response is streamed as Server-Sent Events. |
| `stop` | array | No | `[]` | Stop strings — generation halts when any is produced. |
| `repetition_penalty` | float | No | `1.1` | Repetition penalty. `1.0` = none. |

### Example Request

```bash
curl -X POST http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sharpinfer",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is a transformer model?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

### Non-Streaming Response

```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "model": "sharpinfer",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "A transformer model is a neural network architecture..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 45,
    "total_tokens": 73
  }
}
```

### Streaming Response

See the [Streaming](streaming) page for full SSE format documentation.

---

## List Downloaded Models

```
GET /api/tags
```

Returns all `.gguf` model files in the server's models directory.

**Response:**
```json
{
  "models": [
    {
      "name": "Meta-Llama-3.1-8B-Instruct-Q4_K_M",
      "modified_at": "2026-03-01T12:00:00Z",
      "size": 4815432704,
      "digest": "sha256:abc123...",
      "details": {
        "format": "gguf",
        "family": "",
        "parameter_size": "",
        "quantization_level": ""
      }
    }
  ]
}
```

---

## Pull a Model

```
POST /api/pull
Content-Type: application/json
```

Downloads a model from HuggingFace. Streams download progress as NDJSON.

**Request:**
```json
{
  "name": "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",
  "stream": true
}
```

**NDJSON stream response (one JSON object per line):**
```
{"status":"pulling manifest","digest":"","total":0,"completed":0}
{"status":"downloading","digest":"sha256:abc...","total":4815432704,"completed":524288}
{"status":"downloading","digest":"sha256:abc...","total":4815432704,"completed":1048576}
...
{"status":"success","digest":"","total":0,"completed":0}
```

**Name formats:**

| Format | Example |
|---|---|
| HuggingFace with quant filter | `hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M` |
| HuggingFace latest | `hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:latest` |
| Direct repo (auto-selects) | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` |
| Built-in alias | `llama3`, `mistral`, `phi3`, `gemma`, `codellama`, `qwen2` |

---

## Load a Model

```
POST /api/load
Content-Type: application/json
```

Loads a downloaded model into memory for inference. If another model is loaded, it is unloaded first.

**Request:**
```json
{"name": "Meta-Llama-3.1-8B-Instruct"}
```

**Success response:**
```json
{"status": "success", "model": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"}
```

**Error response (404):**
```json
{"error": "Model not found: Meta-Llama-3.1-8B-Instruct"}
```

The `name` field supports partial matching — `Meta-Llama-3.1-8B` will match `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`.

---

## Show Model Info

```
POST /api/show
Content-Type: application/json
```

Returns metadata for a specific model.

**Request:**
```json
{"name": "Meta-Llama-3.1-8B-Instruct"}
```

**Response:**
```json
{
  "modelfile": "",
  "parameters": "",
  "template": "",
  "details": {
    "format": "gguf",
    "family": "",
    "parameter_size": "",
    "quantization_level": ""
  }
}
```

---

## Delete a Model

```
DELETE /api/delete
Content-Type: application/json
```

Permanently deletes a model file from the models directory.

**Request:**
```json
{"name": "Meta-Llama-3.1-8B-Instruct"}
```

**Success (200):** `{"status": "success"}`

**Not found (404):** `{"error": "Model not found: ..."}`

---

## List Orphaned Models

```
GET /api/orphans
```

Lists model files not referenced by any Modelfile's `FROM` or `ADAPTER` directives. Safe to delete.

**Response:**
```json
{
  "orphaned_count": 2,
  "total_size_bytes": 9630865408,
  "models": [
    {
      "name": "old-experiment-Q4_K_M.gguf",
      "path": "/models/old-experiment-Q4_K_M.gguf",
      "format": "gguf",
      "size_bytes": 4815432704,
      "created_at": "2026-01-15T10:30:00Z"
    }
  ]
}
```

---

## Delete Orphaned Models

```
DELETE /api/orphans
Content-Type: application/json
```

Deletes orphaned model files. Use `dry_run: true` first to preview what would be deleted.

**Request:**
```json
{
  "dry_run": true,
  "models": []
}
```

Pass `"models": ["specific-model.gguf"]` to delete only selected orphans. Leave empty to delete all.

**Response:**
```json
{
  "deleted_count": 2,
  "bytes_freed": 9630865408,
  "deleted_models": ["/models/old-experiment-Q4_K_M.gguf"],
  "errors": [],
  "dry_run": true
}
```

---

## HTTP Status Codes

| Code | Meaning |
|---|---|
| `200` | Success |
| `400` | Bad request — malformed JSON or missing required fields |
| `401` | Unauthorized — API key missing or invalid (enterprise auth only) |
| `404` | Not found — model name not matched |
| `429` | Too many requests — rate limit exceeded (enterprise only) |
| `503` | No model loaded — pull and load a model first |
| `500` | Server error during inference |
