---
license: apache-2.0
base_model: neutronai-code/Quark-12B-A2.5B-v0.8-nes
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
- next-edit-prediction
- neutron-ai
- quark
---

# Quark-12B-A2.5B v0.8-nes — GGUF

GGUF files of [neutronai-code/Quark-12B-A2.5B-v0.8-nes](https://huggingface.co/neutronai-code/Quark-12B-A2.5B-v0.8-nes),
NeutronAI's editor model: **tab completion** and **next-edit prediction** in one set of weights. It's a
mixture-of-experts model with 12B parameters in total, of which only 2.5B are active per token, so it runs fast on a
laptop with a GPU. See the [main model card](https://huggingface.co/neutronai-code/Quark-12B-A2.5B-v0.8-nes) for the
full results and both prompt formats.

**The Q4_K_M file keeps both skills**, measured with llama.cpp on the same benchmarks as the full model:

| | Q4_K_M GGUF | bf16 (full) |
|---|---:|---:|
| Tab completion: pass rate, scored by the repos' own tests (600 holes, 10 repos) | 59.2% | 60.7% |
| Next edit: overall (702 cases, 9 repos) | 46.6% | 46.4% |
| Next edit: exact edit | 28.5% | 28.7% |
| Next edit: left the code alone when no edit was due | 90.7% | 89.7% |

On tab completion, Q4_K_M wins 15 holes and loses 24 against the full model, which is within noise. Both are level
with Qwen2.5-Coder-14B (60.5%), a dense model with about six times the compute per token.

| File | Size | Notes |
|---|---:|---|
| `Quark-12B-A2.5B-v0.8-nes-Q4_K_M.gguf` | 8.1 GB | recommended: runs in 16 GB of RAM; benchmarked above |
| `Quark-12B-A2.5B-v0.8-nes-Q8_0.gguf` | 12.9 GB | near-lossless |

Converted with llama.cpp release b12606 (which supports Mellum2's `mellum` architecture) from the bf16 weights.

## Run with llama.cpp

```bash
llama-server -hf neutronai-code/Quark-12B-A2.5B-v0.8-nes-GGUF:Q4_K_M -c 8192 --port 8080
```

Quark is not a chat model. Send it raw prompts through `/v1/completions` (or `/completion`).

**Tab completion**, in Mellum's suffix-first FIM format:

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

**Next edit:** build the prompt as shown on the
[main model card](https://huggingface.co/neutronai-code/Quark-12B-A2.5B-v0.8-nes#prompt-format-next-edit) and send it
the same way with `"max_tokens": 1024` and `"stop": ["<|region_end|>", "<|endoftext|>"]`. The output is the
rewritten region; if it equals the region you sent, no edit is predicted there.

## License

Apache 2.0, the same as the base model, JetBrains' Mellum2-12B-A2.5B-Base.
