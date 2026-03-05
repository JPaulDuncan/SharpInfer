---
layout: default
title: Interactive Commands
parent: CLI Guide
nav_order: 2
---

# Interactive Commands
{: .no_toc }

Slash commands you can type at the `You >` prompt while the CLI is running.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## General

| Command | Description |
|---|---|
| `/quit` | Exit the CLI. Aliases: `/exit`, `/q` |
| `/reset` | Clear conversation history and the KV cache. Aliases: `/clear` |
| `/config` | Display the current configuration. |
| `/features` | Show which features are active (RAG, LoRA, speculative, etc.). |
| `/save-config <path>` | Save current settings to a JSON config file. |
| `/hardware` | List detected hardware backends. Alias: `/hw` |
| `/export-modelfile <path>` | Export current settings as a Modelfile. |

---

## Generation Tuning

You can adjust generation parameters on the fly without restarting the CLI.

| Command | Example | Description |
|---|---|---|
| `/temp <value>` | `/temp 0.3` | Set sampling temperature. |
| `/topp <value>` | `/topp 0.95` | Set nucleus sampling top-p. |
| `/topk <value>` | `/topk 20` | Set top-k token filter. |
| `/cfg <scale>` | `/cfg 1.5` | Enable classifier-free guidance. Use `0` or `1` to disable. |
| `/consistency <n>` | `/consistency 5` | Enable self-consistency with N candidates. Use `off` to disable. |
| `/beam <n>` | `/beam 4` | Enable beam search with N beams. Use `off` to disable. |

---

## LoRA Adapters

| Command | Description |
|---|---|
| `/lora list` | Show all loaded adapters and which is active. |
| `/lora apply <name>` | Activate a named adapter. |
| `/lora remove <name>` | Deactivate an adapter (revert to base model). |

---

## RAG (Retrieval-Augmented Generation)

| Command | Description |
|---|---|
| `/rag add <file>` | Ingest a document or directory into the knowledge base. |
| `/rag status` | Show RAG stats: chunks indexed, memory used. |
| `/rag on` | Enable RAG augmentation for subsequent responses. |
| `/rag off` | Disable RAG augmentation (keeps the index, just stops using it). |
| `/rag clear` | Remove all documents from the knowledge base. |

---

## Model Management

| Command | Description |
|---|---|
| `/model list` | Show models available in the local registry. |
| `/pull <name>` | Download a model from HuggingFace. |
| `/pull aliases` | List built-in model aliases (e.g., `llama3`, `mistral`). |
| `/pull list` | Show downloaded models. |
| `/pull delete <name>` | Remove a downloaded model. |
| `/quantize info` | Show the current model's quantization type. |

---

## Agents and MCP

| Command | Description |
|---|---|
| `/agent list` | Show configured agent flows. |
| `/agent run <flow> <input>` | Execute a named agent flow with the given input. |
| `/mcp list` | Show connected MCP servers and their available tools. |
| `/mcp connect <url>` | Connect to an MCP server at runtime. |

---

## Tips

- Use `/features` after startup to confirm which optional features are enabled.
- Use `/hardware` to confirm your GPU was detected — it shows backend priority order.
- After adjusting parameters with `/temp`, `/topp`, etc., the change takes effect immediately on the next message.
- `/reset` is useful if the model starts going off-topic — it clears all history and lets you start fresh without reloading the model.
