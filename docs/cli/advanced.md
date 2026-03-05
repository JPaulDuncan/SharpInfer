---
layout: default
title: Advanced Features
parent: CLI Guide
nav_order: 7
---

# Advanced Features
{: .no_toc }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Speculative Decoding

Speculative decoding uses a small, fast "draft" model to predict multiple tokens ahead, then verifies them in parallel with the main model. Valid predictions are accepted; incorrect ones are discarded and regeneration resumes from the main model. This typically yields **2–3× faster generation** with no quality loss.

```bash
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/llama-3.1-8b.gguf \
  --speculative \
  --draft-model ./models/llama-3.2-1b.gguf
```

**Best draft models:** A 1B or 3B model from the same family as your main model. For LLaMA 3.1 8B, use LLaMA 3.2 1B. For Mistral 7B, use Mistral 3B.

---

## LoRA Adapters

LoRA (Low-Rank Adaptation) adapters customize model behavior by modifying specific attention and MLP layers. They're small files (tens to hundreds of MB) that sit on top of a base model.

**Load at startup:**

```bash
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/llama-3.1-8b.gguf \
  --lora "code:./adapters/code-assist.bin" \
  --lora "formal:./adapters/formal-tone.bin"
```

**Manage at runtime:**

```
/lora list              # See all adapters
/lora apply code        # Activate the "code" adapter
/lora remove code       # Return to the base model
```

SharpInfer loads PEFT-format adapters exported from HuggingFace Transformers.

---

## Prompt Caching

Prompt caching serializes the KV state after processing a system prompt and saves it to disk. On subsequent sessions with the same prompt, the saved state is loaded directly — skipping the encoding step entirely.

```bash
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/llama-3.1-8b.gguf \
  --prompt-cache \
  --system "You are a senior .NET engineer. Help the user write idiomatic C# code."
```

Cache files are stored in `~/.sharpinfer/cache/` (or the path set in the config file) and keyed by a hash of the model path and system prompt. You'll notice the first session is normal speed, and subsequent sessions start immediately.

---

## RAG (Retrieval-Augmented Generation)

RAG grounds model responses in your own documents. At query time, relevant passages are retrieved and injected into the prompt context.

```bash
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/llama-3.1-8b.gguf \
  --rag \
  --rag-docs ./docs/
```

**At runtime:**

```
/rag add ./additional-doc.pdf    # Add more documents
/rag status                      # Show how many chunks are indexed
/rag off                         # Temporarily disable RAG
/rag on                          # Re-enable
```

Documents are chunked, embedded, and stored in an in-memory vector index. Supported file types: `.pdf`, `.txt`, `.md`, `.docx`, `.html`.

---

## Multi-Agent Orchestration

Define multiple AI agents and orchestrate them in flows. Configure agents in the `multiAgent` section of your config, then run flows from the CLI:

```
/agent list                    # See configured flows
/agent run research-and-write "Explain quantum key distribution"
```

Available flow types:

| Flow Type | Description |
|---|---|
| `chain` | Sequential: output from agent A becomes input to agent B |
| `parallel` | Concurrent: all agents process the input, results are synthesized |
| `debate` | Adversarial: agents argue positions, optional judge picks winner |
| `router` | Conditional: input is analyzed and routed to the best-fit agent |
| `handoff` | Collaborative: agents decide when to pass work to another agent |
| `review-loop` | Iterative: a writer agent produces output, a reviewer critiques it |

See the [Configuration File]({% link cli/configuration.md %}) page for the full `multiAgent` config schema.

---

## Tool Calling

Enable tool calling to let the model use web search and URL reading:

```bash
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/llama-3.1-8b.gguf \
  --tools
```

Configure a search API key in the config file for web search. Without a key, URL reading still works.

---

## MCP (Model Context Protocol)

Connect to MCP tool servers to extend the model's capabilities with filesystem access, database queries, custom APIs, and more:

```
/mcp connect http://localhost:3001    # Connect to an MCP server
/mcp list                             # See available tools
```

Configure persistent MCP servers in the `mcpServers` section of your config file.

---

## Structured Output

Force the model to produce output conforming to a JSON schema:

```bash
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/llama-3.1-8b.gguf \
  --structured-output
```

Define the target schema in the config file's `structuredOutput` section. Useful for data extraction, classification, and pipelines that consume model output programmatically.

---

## Multimodal Vision

With a vision-capable base model and a configured image encoder, SharpInfer can analyze images alongside text:

```bash
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/llava-1.6.gguf \
  --multimodal
```

Include image file paths in your messages using the format the model expects. CLIP (ViT-L/14) and SigLIP encoders are supported.
