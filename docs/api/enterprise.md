---
layout: default
title: Enterprise Middleware
parent: API Reference
nav_order: 4
---

# Enterprise Middleware
{: .no_toc }

Authentication, rate limiting, audit logging, and usage metering for production deployments.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

{: .note }
Enterprise features are **disabled by default**. Enable them individually with command-line flags or in the `enterprise` section of your config file.

---

## API Key Authentication

When enabled, all API requests must include a valid API key in the `Authorization` header.

### Enabling

```bash
dotnet run --project src/SharpInfer.Api -- --model model.gguf --auth
```

Or in config:
```json
"enterprise": {
  "auth": { "enabled": true }
}
```

### Using Keys

All requests must include:

```
Authorization: Bearer sk-your-api-key-here
```

Requests without a valid key receive `401 Unauthorized`.

### Key Storage

API keys are SHA-256 hashed before storage — the server never stores plaintext keys. Manage keys through the enterprise management API endpoints.

---

## Rate Limiting

Limits requests per API key using a **token bucket** algorithm. Each key gets a configurable number of request tokens per second with a burst capacity.

### Enabling

```bash
dotnet run --project src/SharpInfer.Api -- --model model.gguf --rate-limit
```

Or in config:
```json
"enterprise": {
  "rateLimit": {
    "enabled": true,
    "tokensPerSecond": 10,
    "burstCapacity": 50
  }
}
```

### Behavior

When a key exhausts its token bucket, requests receive:

```
HTTP 429 Too Many Requests
Retry-After: 2
```

The `Retry-After` header indicates seconds until capacity is available.

---

## Audit Logging

Records every API request to a JSONL file for compliance, debugging, and analysis.

### Enabling

```bash
dotnet run --project src/SharpInfer.Api -- --model model.gguf --audit
```

Or in config:
```json
"enterprise": {
  "audit": {
    "enabled": true,
    "logPath": "./logs/audit.jsonl",
    "maxSizeBytes": 104857600
  }
}
```

### Log Entry Format

Each line is a JSON object:
```json
{
  "timestamp": "2026-03-01T12:00:00.123Z",
  "api_key_hash": "sha256:abc123...",
  "endpoint": "/v1/chat/completions",
  "method": "POST",
  "request_bytes": 284,
  "response_time_ms": 1240,
  "status_code": 200,
  "tokens_generated": 45
}
```

Logs rotate automatically when the file reaches `maxSizeBytes` (default: 100 MB).

---

## Usage Metering

Tracks token consumption per API key with daily breakdowns.

### Enabling

```bash
dotnet run --project src/SharpInfer.Api -- --model model.gguf --metering
```

Or in config:
```json
"enterprise": {
  "metering": { "enabled": true }
}
```

### Usage Data

Metering captures per-key, per-day:
- Prompt tokens used
- Completion tokens generated
- Total tokens
- Estimated cost (based on configurable per-token pricing)

Access usage reports through the enterprise management API.

---

## Enabling Multiple Features Together

```bash
dotnet run --project src/SharpInfer.Api -- \
  --model ./models/llama-3.1-8b.gguf \
  --auth \
  --rate-limit \
  --audit \
  --metering \
  --port 3512
```

Or in a config file for cleaner management:

```json
{
  "model": { "path": "./models/llama-3.1-8b.gguf" },
  "enterprise": {
    "auth": { "enabled": true },
    "rateLimit": {
      "enabled": true,
      "tokensPerSecond": 20,
      "burstCapacity": 100
    },
    "audit": {
      "enabled": true,
      "logPath": "./logs/audit.jsonl"
    },
    "metering": { "enabled": true }
  }
}
```
