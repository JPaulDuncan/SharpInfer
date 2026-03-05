---
layout: default
title: Integration Guide
nav_order: 5
has_children: true
permalink: /integration/
---

# Integration Guide
{: .no_toc }

Connect SharpInfer to your tools, applications, and frontends.
{: .fs-6 .fw-300 }

---

## Overview

SharpInfer's OpenAI-compatible API means any tool that targets the OpenAI Chat Completions format works with a single URL change. No API key is required by default.

**API base URL:** `http://localhost:3512/v1`

---

## Quick Connection Reference

### Continue.dev (VS Code AI Coding)

In `.continue/config.json`:
```json
{
  "models": [{
    "title": "SharpInfer (local)",
    "provider": "openai",
    "model": "sharpinfer",
    "apiBase": "http://localhost:3512/v1"
  }]
}
```

### Open WebUI

In Settings → Connections → OpenAI API:
- **URL:** `http://localhost:3512/v1`
- **API Key:** `none` (any value works)

### Python (openai library)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:3512/v1", api_key="not-needed")
```

### JavaScript (fetch)

```javascript
const res = await fetch("http://localhost:3512/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ model: "sharpinfer", messages: [...] })
});
```

### curl

```bash
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sharpinfer","messages":[{"role":"user","content":"Hi"}]}'
```

---

## In This Section

- **[Third-Party Tools]({% link integration/third-party.md %})** — Continue.dev, Open WebUI, LangChain, LlamaIndex step-by-step setup
- **[Code Examples]({% link integration/code-examples.md %})** — Complete Python, JavaScript/TypeScript, and curl examples with streaming and non-streaming patterns
- **[Frontend Configuration]({% link integration/frontend-config.md %})** — Complete checklist and differences from the OpenAI API
