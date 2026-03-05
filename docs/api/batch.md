---
layout: default
title: Batch Processing
parent: API Reference
nav_order: 3
---

# Batch Processing
{: .no_toc }

Submit many prompts at once and retrieve results asynchronously.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Overview

The batch API is for offline workloads where you don't need real-time responses: processing datasets, running evaluations, generating content in bulk, or any task where you want to submit work and pick up results later.

Enable batch processing with the `--batch-processing` flag or in the config file.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/batches` | Submit a new batch job |
| `GET` | `/v1/batches` | List all batch jobs |
| `GET` | `/v1/batches/{id}` | Get status and progress |
| `GET` | `/v1/batches/{id}/results` | Retrieve completed results |
| `POST` | `/v1/batches/{id}/cancel` | Cancel a running batch |
| `DELETE` | `/v1/batches/{id}` | Delete a completed batch |

---

## Submitting a Batch

```bash
curl -X POST http://localhost:3512/v1/batches \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"custom_id": "q1", "prompt": "What is Python?"},
      {"custom_id": "q2", "prompt": "What is C#?"},
      {"custom_id": "q3", "prompt": "What is Rust?"}
    ],
    "priority": "normal",
    "default_max_tokens": 256,
    "default_temperature": 0.7,
    "webhook_url": "https://myapp.com/webhook/batch-complete"
  }'
```

**Request Fields**

| Field | Type | Description |
|---|---|---|
| `requests` | array | Array of request items. Each item has: `custom_id` (your identifier), `prompt` or `messages`, and optional per-item `max_tokens` and `temperature`. |
| `priority` | string | `low`, `normal`, or `high`. High-priority batches jump ahead in the queue. Default: `normal`. |
| `default_max_tokens` | integer | Default max tokens for all requests. Can be overridden per item. Default: `512`. |
| `default_temperature` | float | Default temperature. Can be overridden per item. Default: `0.7`. |
| `webhook_url` | string | Optional URL to POST when the batch completes. |
| `metadata` | object | Arbitrary key-value pairs attached to the batch job. |

**Response:**
```json
{
  "id": "batch-abc123",
  "status": "pending",
  "created_at": "2026-03-01T12:00:00Z",
  "total_requests": 3
}
```

---

## Checking Status

```bash
curl http://localhost:3512/v1/batches/batch-abc123
```

**Response:**
```json
{
  "id": "batch-abc123",
  "status": "running",
  "progress": 67,
  "completed": 2,
  "total": 3,
  "created_at": "2026-03-01T12:00:00Z",
  "started_at": "2026-03-01T12:00:05Z"
}
```

Status values: `pending`, `running`, `completed`, `failed`, `cancelled`.

---

## Retrieving Results

```bash
curl http://localhost:3512/v1/batches/batch-abc123/results
```

**Response:**
```json
{
  "id": "batch-abc123",
  "status": "completed",
  "results": [
    {
      "custom_id": "q1",
      "status": "success",
      "output": "Python is a high-level, interpreted programming language..."
    },
    {
      "custom_id": "q2",
      "status": "success",
      "output": "C# is a modern, object-oriented language developed by Microsoft..."
    },
    {
      "custom_id": "q3",
      "status": "success",
      "output": "Rust is a systems programming language focused on safety and performance..."
    }
  ]
}
```

---

## Chat Message Requests

Items in a batch can use `messages` instead of `prompt` for multi-turn context:

```json
{
  "requests": [
    {
      "custom_id": "chat-1",
      "messages": [
        {"role": "system", "content": "You are a concise technical explainer."},
        {"role": "user", "content": "What is a B-tree?"}
      ],
      "max_tokens": 128
    }
  ]
}
```

---

## Cancelling a Batch

```bash
curl -X POST http://localhost:3512/v1/batches/batch-abc123/cancel
```

Completed items up to the cancellation point are available in the results.

---

## Batch Features

**Priority queuing** — Use `"priority": "high"` for time-sensitive batches. They jump ahead of normal and low-priority jobs.

**Automatic retry** — Failed items are retried up to 2 times automatically before being marked as failed.

**Webhook callbacks** — Provide a `webhook_url` and SharpInfer will POST to it when the batch completes, including the batch ID and final status.

**Per-item settings** — Each request in the batch can override `max_tokens` and `temperature` independently.
