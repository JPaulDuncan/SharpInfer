---
layout: default
title: Third-Party Tools
parent: Integration Guide
nav_order: 1
---

# Third-Party Tools
{: .no_toc }

Step-by-step setup for popular AI tools and frameworks.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Continue.dev (VS Code)

Continue.dev is a popular AI coding assistant extension for VS Code and JetBrains IDEs.

1. Install **Continue** from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=Continue.continue).
2. Open the Continue panel and click the gear icon to open settings.
3. Add SharpInfer as a model provider in `.continue/config.json`:

```json
{
  "models": [
    {
      "title": "SharpInfer (local)",
      "provider": "openai",
      "model": "sharpinfer",
      "apiBase": "http://localhost:3512/v1",
      "apiKey": "none"
    }
  ],
  "tabAutocompleteModel": {
    "title": "SharpInfer Autocomplete",
    "provider": "openai",
    "model": "sharpinfer",
    "apiBase": "http://localhost:3512/v1",
    "apiKey": "none"
  }
}
```

4. Select "SharpInfer (local)" from the model dropdown in the Continue panel.

{: .note }
For best code completion results, load a code-focused model like CodeLlama 7B or DeepSeek Coder 6.7B.

---

## Open WebUI

Open WebUI provides a ChatGPT-like interface for local models.

**With Docker:**
```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:3512/v1 \
  -e OPENAI_API_KEY=none \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

Open [http://localhost:3000](http://localhost:3000) and SharpInfer appears as your model provider.

**Manual setup:**
1. Start Open WebUI
2. Go to **Settings → Connections**
3. Set **OpenAI API URL** to `http://localhost:3512/v1`
4. Set **API Key** to any non-empty string (e.g., `none`)
5. Save and reload — your model appears in the model selector

---

## LangChain (Python)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:3512/v1",
    api_key="not-needed",
    model="sharpinfer",
    temperature=0.7
)

response = llm.invoke("Explain the CAP theorem in simple terms.")
print(response.content)
```

**Streaming with LangChain:**
```python
for chunk in llm.stream("Write a haiku about programming."):
    print(chunk.content, end="", flush=True)
```

---

## LlamaIndex (Python)

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    api_base="http://localhost:3512/v1",
    api_key="not-needed",
    model="sharpinfer"
)

from llama_index.core import Settings
Settings.llm = llm

response = llm.complete("What is a vector database?")
print(response.text)
```

---

## Ollama-Compatible Clients

SharpInfer uses a similar model management API to Ollama (`/api/pull`, `/api/tags`, etc.). Many Ollama-compatible clients can point to SharpInfer directly by changing the base URL to `http://localhost:3512`.

---

## Direct API (curl)

```bash
# Non-streaming
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sharpinfer",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7
  }'

# Streaming
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sharpinfer","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'
```
