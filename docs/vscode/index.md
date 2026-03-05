---
layout: default
title: VS Code Extension
nav_order: 6
has_children: true
permalink: /vscode/
---

# VS Code Extension
{: .no_toc }

Local AI code completion and chat inside Visual Studio Code — no cloud required.
{: .fs-6 .fw-300 }

---

## Overview

The SharpInfer VS Code Extension provides:

- **Inline code completion** — Ghost-text suggestions as you type, powered by Fill-in-the-Middle (FIM) prompting
- **Text generation** — Write documentation, comments, and boilerplate on demand
- **Multi-turn chat** — Ask questions about code, debug issues, and brainstorm solutions in a side panel

Everything runs locally via the `SharpInfer.VsCode` backend process. Your code never leaves your machine.

---

## Architecture

```
VS Code (TypeScript Extension)
      │
      │  stdin/stdout  (JSON-RPC 2.0 with LSP content-length framing)
      │
SharpInfer.VsCode  (.NET 8 process)
      │
      │  Direct library calls
      │
SharpInfer.Core  (Inference Engine)
      │
CPU / GPU  (CUDA · Metal · Vulkan · ARM NEON)
```

The TypeScript extension spawns the .NET backend process when VS Code activates. Communication uses LSP-style JSON-RPC over stdin/stdout — the same protocol used by language servers.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Visual Studio Code | v1.80 or later |
| .NET 8.0 Runtime | Required to run the backend process |
| Model file | A GGUF or Safetensors model — code-specific models work best |
| RAM | 8 GB minimum; 16 GB for larger models |
| GPU *(optional)* | CUDA, Metal, or Vulkan for faster completions |

{: .note }
For code completion, smaller models (1–7B) respond faster and produce focused suggestions. Use a larger model (13B+) only if you have the hardware and need complex reasoning.

---

## In This Section

- **[Setup]({% link vscode/setup.md %})** — Build the backend and configure the extension
- **[Code Completion]({% link vscode/completion.md %})** — How inline completions work, FIM prompting, settings
- **[Chat]({% link vscode/chat.md %})** — Using the chat panel for coding assistance
- **[JSON-RPC Protocol]({% link vscode/protocol.md %})** — Full protocol reference for extension developers
