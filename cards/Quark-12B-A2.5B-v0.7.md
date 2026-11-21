---
license: apache-2.0
base_model: JetBrains/Mellum2-12B-A2.5B-Base
library_name: transformers
pipeline_tag: text-generation
language:
- code
tags:
- code
- fill-in-the-middle
- code-completion
- tab-completion
- mixture-of-experts
- neutron-ai
- quark
---

# Quark-12B-A2.5B v0.7

**Quark** is NeutronAI's tab-completion model: you give it the code before and after the cursor, and it writes the
missing middle. It is a fine-tune of JetBrains' [Mellum2-12B-A2.5B-Base](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Base),
a mixture-of-experts model with 12B parameters in total but only **2.5B active per token**, so it runs about as fast
as a 2.5B model. On our test-based benchmark it matches Qwen2.5-Coder-14B, a dense model with about six times the
compute per token.

- **Languages:** Python, TypeScript, JavaScript, C++
- **Task:** fill-in-the-middle (FIM) with optional repository context (signatures from the repo's other files)
- **License:** Apache 2.0, the same as the base model
- **GGUF:** [neutronai-code/Quark-12B-A2.5B-v0.7-GGUF](https://huggingface.co/neutronai-code/Quark-12B-A2.5B-v0.7-GGUF)

## Results: scored by real tests, not text matching

Most FIM benchmarks check whether the completion matches the original text. Ours checks whether **the code still
works**. Each completion is spliced into the real repository, and the repository's own test suite decides pass or fail.

**Benchmark:** 600 holes (short and long) from 10 recent GitHub repositories that no model here was trained on. The
prompt includes about 4k characters of repository context. Every model gets the same prompts, with greedy decoding.

| Model | Active params | Pass rate | Python | TypeScript | JavaScript | C++ | Short holes | Long holes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-14B (base) | 14.7B | 60.5% | 68.9 | 58.9 | 47.5 | 63.3 | 71.6 | 46.4 |
| **Quark-12B-A2.5B v0.7** | **2.5B** | **59.8%** | 67.2 | 57.2 | 46.7 | 65.8 | 70.4 | 46.4 |
| Quark-12B-A2.5B v0.6 | 2.5B | 59.8% | 69.4 | 56.7 | 44.2 | 65.8 | 69.9 | 47.2 |
| Qwen2.5-Coder-7B (base) | 7.6B | 54.7% | 62.2 | 56.7 | 38.3 | 56.7 | 63.0 | 44.2 |
| Mellum2-12B-A2.5B-Base (our base) | 2.5B | 54.5% | 64.4 | 54.4 | 35.0 | 59.2 | 63.6 | 43.0 |
| Quark-1.5B v0.5 | 1.5B | 52.2% | 65.6 | 49.4 | 38.3 | 50.0 | 62.4 | 39.2 |
| DeepSeek-Coder-V2-Lite-Base | 2.4B | 47.0% | 53.3 | 50.0 | 38.3 | 41.7 | 58.2 | 32.8 |
| Qwen2.5-Coder-3B (base) | 3.1B | 43.8% | 50.6 | 43.9 | 29.2 | 48.3 | 52.5 | 32.8 |
| StarCoder2-3B | 3B | 42.5% | 43.9 | 47.2 | 32.5 | 43.3 | 51.9 | 30.6 |
| Qwen2.5-Coder-1.5B (base) | 1.5B | 36.3% | 47.2 | 37.8 | 24.2 | 30.0 | 43.9 | 26.8 |
| Mellum-4b-base | 4B | 35.8% | 40.6 | 38.9 | 25.0 | 35.0 | 45.1 | 24.2 |

- **Quark matches a 14B dense model at about a sixth of the compute per token.** Qwen2.5-Coder-14B is 0.7 points
  ahead, but head to head the two are even: v0.7 wins 56 holes and loses 60. Quark is ahead on C++ and ties on long
  holes.
- Against DeepSeek-Coder-V2-Lite, a mixture-of-experts model of about the same active size (16B total, 2.4B active),
  v0.7 is 12.8 points ahead.
- **The GGUF keeps the score.** The Q4_K_M [GGUF](https://huggingface.co/neutronai-code/Quark-12B-A2.5B-v0.7-GGUF)
  (8.1 GB) passes 60.8% on the same benchmark; against these bf16 weights it wins 28 holes and loses 22, within noise.
- Against its own base model, v0.7 wins 65 holes and loses 33. Against Qwen2.5-Coder-7B it wins 81 and loses 50.
- Codestral-22B-v0.1 is not in the table. With raw FIM prompts its completions usually ran past the end of the hole
  (median 1,372 characters, against 83 for Quark), so its score (25.3%) says more about our setup than the model.
- **Second check on unseen code:** on 1,126 more holes from 21 other repositories Quark was never trained on, v0.7
  passes **58.9%**, close to its benchmark score.
- v0.7 trained for twice as long as v0.6 and ties it, so this is the plateau for this recipe. Doubling the repository
  context to 8k characters also gives no gain (60.2%).

## Prompt format

Quark uses Mellum's FIM tokens with the **suffix first**:

```
<fim_suffix>{code after the cursor}<fim_prefix>{code before the cursor}<fim_middle>
```

With repository context, put each related file first as `<filename>{path}\n{its signatures}\n`, then the current
file's path, then the FIM part:

```
<filename>src/rates.py
def tax_multiplier(region: str) -> float:
class Invoice:
    def total(self) -> float:
<filename>src/billing.py
<fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>
```

The context Quark was trained on is one-line signatures (functions, classes, members, constants) from the repo's other
files, with bodies cut to `{...}`, ordered most relevant first. The model then writes the middle and ends with
`<|endoftext|>`. Also stop on `<filename>`, `<fim_prefix>` and `<fim_suffix>`.

```python
def quark_prompt(prefix: str, suffix: str, path: str = "", context: list[tuple[str, str]] = ()) -> str:
    """context: (path, signatures) pairs for related files, most relevant first."""
    ctx = "".join(f"<filename>{p}\n{text.rstrip()}\n" for p, text in context)
    head = f"{ctx}<filename>{path}\n" if ctx else ""
    return f"{head}<fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>"
```

## Usage

### vLLM (what we benchmark with)

```bash
vllm serve neutronai-code/Quark-12B-A2.5B-v0.7 --max-model-len 8192
```

```python
import requests

prompt = quark_prompt(prefix="def is_even(n: int) -> bool:\n    return ", suffix="\n\nprint(is_even(4))\n")
r = requests.post("http://localhost:8000/v1/completions", json={
    "model": "neutronai-code/Quark-12B-A2.5B-v0.7", "prompt": prompt, "max_tokens": 256, "temperature": 0,
    "stop": ["<|endoftext|>", "<filename>", "<fim_prefix>", "<fim_suffix>"], "skip_special_tokens": False,
})
print(r.json()["choices"][0]["text"])
```

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "neutronai-code/Quark-12B-A2.5B-v0.7"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype="bfloat16", device_map="auto")
ids = tok(quark_prompt("def is_even(n: int) -> bool:\n    return ", "\n"), return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

The bf16 weights are 24 GB. For laptops, use the [GGUF](https://huggingface.co/neutronai-code/Quark-12B-A2.5B-v0.7-GGUF).

## Training

- **Method:** supervised FIM fine-tuning with LoRA (rank 32, on the attention layers and on all 64 experts; 435M
  trainable parameters), merged into the base weights.
- **Data:** about 179k FIM examples from 265 open-source repositories under permissive licenses (MIT, Apache 2.0,
  BSD, ISC). Python, TypeScript, JavaScript and C++, about evenly split.
  - Hole types: whole statements and blocks (short and long), mid-line completions, random spans, holes where the
    right answer is empty, and whole-file completion.
  - Half of the examples carry repository context in the format above.
  - The benchmark repositories are excluded, and their code was checked against the training data.
- **Two stages:**
  - Stage 1 (v0.6): one epoch over 59.6k examples.
  - Stage 2 (v0.7): one epoch over 119.3k new examples, continued from v0.6.
  - Both: learning rate 1e-4, effective batch size 16, sequences up to 4,096 tokens.
- **Hardware:** NVIDIA A100 80GB and RTX Pro 6000 (96GB).

## Limitations

- Long holes (a whole function body or block) pass about 46% of the time; short ones about 70%.
- JavaScript is the weakest language (47%).
- When the code needed comes from a repository name that isn't in the prompt's context, Quark usually can't guess it.
  Give it good context.
- It is a completion model, not a chat model: it doesn't follow instructions.
- Like any code model, it can write code that looks right but is wrong. Review what you accept.

## About NeutronAI

NeutronAI builds open coding models. **Quark** is the fast tab-completion line; **Hadron** is the reasoning and
repository-editing line. Code: [github.com/neutronai-code/neutron-ai](https://github.com/neutronai-code/neutron-ai).
