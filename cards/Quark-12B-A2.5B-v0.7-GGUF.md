---
license: apache-2.0
base_model: neutronai-code/Quark-12B-A2.5B-v0.7
base_model_relation: quantized
language:
- code
tags:
- gguf
- llama.cpp
- code
- fill-in-the-middle
- code-completion
- tab-completion
- neutron-ai
- quark
---

# Quark-12B-A2.5B v0.7 — GGUF

GGUF files of [neutronai-code/Quark-12B-A2.5B-v0.7](https://huggingface.co/neutronai-code/Quark-12B-A2.5B-v0.7),
NeutronAI's tab-completion model. It's a mixture-of-experts model with 12B parameters in total, of which only 2.5B are
active per token, so it runs fast on a laptop. See the
[main model card](https://huggingface.co/neutronai-code/Quark-12B-A2.5B-v0.7) for the full results.

**Quantizing costs nothing measurable.** On our FIM benchmark, where each completion is scored by the repository's own
tests (600 holes from 10 repositories), the Q4_K_M file passes **60.8%**. The full bf16 model passes 59.8%: head to
head, Q4_K_M wins 28 holes and loses 22, which is within noise. Both are level with Qwen2.5-Coder-14B (60.5%), a dense
model with about six times the compute per token.

| File | Size | Notes |
|---|---:|---|
| `Quark-12B-A2.5B-v0.7-Q4_K_M.gguf` | 8.1 GB | recommended: runs in 16 GB of RAM; benchmarked above |
| `Quark-12B-A2.5B-v0.7-Q8_0.gguf` | 12.9 GB | near-lossless |

Converted with llama.cpp release b12606 (which supports Mellum2's `mellum` architecture) from the bf16 weights.

## Run with llama.cpp

```bash
llama-server -hf neutronai-code/Quark-12B-A2.5B-v0.7-GGUF:Q4_K_M -c 8192 --port 8080
```

Quark is a **fill-in-the-middle** model, not a chat model. Send it raw prompts in Mellum's suffix-first format:

```python
import requests

def quark_prompt(prefix: str, suffix: str, path: str = "", context: list[tuple[str, str]] = ()) -> str:
    """context: (path, signatures) pairs for related files in the repo, most relevant first."""
    ctx = "".join(f"<filename>{p}\n{text.rstrip()}\n" for p, text in context)
    head = f"{ctx}<filename>{path}\n" if ctx else ""
    return f"{head}<fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>"

r = requests.post("http://localhost:8080/v1/completions", json={
    "prompt": quark_prompt("def is_even(n: int) -> bool:\n    return ", "\n\nprint(is_even(4))\n"),
    "max_tokens": 256, "temperature": 0,
    "stop": ["<|endoftext|>", "<filename>", "<fim_prefix>", "<fim_suffix>"],
})
print(r.json()["choices"][0]["text"])
```

llama.cpp's `/infill` endpoint builds the prompt in a different order (prefix first). Quark was trained suffix first,
so use `/v1/completions` with the prompt above.

## License

Apache 2.0, the same as the base model, JetBrains' Mellum2-12B-A2.5B-Base.
