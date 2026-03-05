---
layout: default
title: Generation Parameters
parent: CLI Guide
nav_order: 5
---

# Generation Parameters
{: .no_toc }

Understanding the controls that shape the model's output.
{: .fs-6 .fw-300 }

---

## Temperature

Temperature controls the randomness of token selection at each step.

- **`0.0`** — The model always picks the single most probable next token. Completely deterministic and repetitive.
- **`0.1–0.3`** — Very focused. Good for factual Q&A, code generation, and data extraction tasks where you need predictable output.
- **`0.5–0.7`** — Balanced. The default `0.7` works well for most conversational use cases.
- **`0.8–1.0`** — Creative. Better for brainstorming, story writing, and exploratory tasks.
- **`> 1.0`** — Very creative, but can produce incoherent or nonsensical text.

```bash
/temp 0.3      # Focused, factual
/temp 0.8      # Creative writing
```

---

## Top-P (Nucleus Sampling)

Top-P limits token selection to the smallest set of tokens whose combined probability reaches the threshold.

With `top_p = 0.9`, the model considers only tokens that together account for 90% of the probability mass. This cuts off unlikely tokens that temperature alone might occasionally pick.

- **`0.9`** — Good default. Keeps responses natural while cutting off extremes.
- **`0.95`** — Slightly more variety.
- **`1.0`** — Effectively disabled — all tokens are eligible.
- **`< 0.8`** — Very conservative. Can feel repetitive.

Top-P and temperature work together. Both are applied — top-P first, then temperature within the surviving set.

---

## Top-K

Top-K simply limits the model to selecting among the K most probable tokens at each step. It's less nuanced than top-P but fast and simple.

- **`40`** — Default. Adequate for most use cases.
- **`1`** — Greedy sampling (always pick the best token). Similar to `temperature = 0.0`.
- **`100+`** — More variety.

```bash
/topk 20       # More focused
/topk 100      # More variety
```

---

## Repetition Penalty

Penalizes tokens that have already appeared in the context, discouraging the model from repeating itself.

- **`1.0`** — No penalty. The model may repeat phrases freely.
- **`1.05–1.15`** — Light penalty. Reduces obvious repetition without distorting style. Good default range.
- **`1.2–1.3`** — Stronger penalty. Useful for long-form generation or summaries.
- **`> 1.4`** — Can cause strange word choices or refusal to use common words.

---

## Max Tokens

The maximum number of tokens the model will generate per response. One token is roughly 0.75 words in English.

| Use case | Recommended value |
|---|---|
| Short Q&A | `128–256` |
| Conversational chat | `512` (default) |
| Long explanations | `1024–2048` |
| Document/essay generation | `2048–4096` |

{: .note }
Generating more tokens is slower. Start with the default and increase if you find responses getting cut off.

---

## Seed

Setting a seed makes generation fully reproducible. Given the same prompt and the same seed, the model will produce the exact same output every time.

```bash
dotnet run --project src/SharpInfer.Cli -- \
  --model ./models/model.gguf --seed 42 --temp 0.7
```

Useful for testing, benchmarking, and debugging prompt variations.

---

## Recommended Presets

| Task | `--temp` | `--top-p` | `--top-k` | `--repeat-penalty` |
|---|---|---|---|---|
| Factual Q&A | `0.1` | `0.9` | `20` | `1.05` |
| Code generation | `0.2` | `0.95` | `40` | `1.1` |
| Conversation | `0.7` | `0.9` | `40` | `1.1` |
| Creative writing | `0.85` | `0.95` | `80` | `1.05` |
| Brainstorming | `1.0` | `0.95` | `100` | `1.0` |
