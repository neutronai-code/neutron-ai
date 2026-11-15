"""NeutronAI Release Publisher (`release.py`) — Hugging Face Model Card + Leaderboard Table + Ollama Modelfile Generator.

Why open-weight coding startups blow up on Hugging Face / GitHub / X (and unlock $100k-$350k GPU grants):
  - Shipping clean reproducibility artifacts: a transparent benchmark comparison table (Base vs. Hadron / Quark),
    contaminant-free audit stats, exact system prompts, and a 1-command local `Modelfile` so anyone with Ollama
    can run `ollama create hadron-7b -f Modelfile` in 10 seconds.

Usage:
  uv run release.py --self-check
  uv run release.py --model-name Hadron-7B-v0.9 --base-model Qwen/Qwen2.5-Coder-7B-Instruct --eval-report eval_report.jsonl --base-report base_report.jsonl
"""
import argparse, collections, json, os


def build_release_bundle(
    model_name: str,
    base_model: str,
    gguf_path: str,
    eval_rows: list[dict],
    base_rows: list[dict] | None = None,
    audit_stats: dict | None = None,
) -> tuple[str, str]:
    """Return `(huggingface_readme_md, ollama_modelfile)`."""
    base_map = {r["id"]: r for r in (base_rows or [])}
    by_cat = collections.defaultdict(list)
    for r in eval_rows:
        by_cat[r.get("category", "general")].append(r)

    overall = sum(r.get("pass@1", 0.0) for r in eval_rows) / max(1, len(eval_rows))
    # Average base pass@1 over the tasks both reports share (dividing by len(base_map) understated the base
    # whenever the base report had tasks the new eval lacks, inflating the published delta).
    shared = [base_map[r["id"]].get("pass@1", 0.0) for r in eval_rows if r["id"] in base_map]
    base_overall = sum(shared) / len(shared) if shared else None

    table_lines = [
        "| Category | Tasks | Base `pass@1` | **" + model_name + "** `pass@1` | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat, items in sorted(by_cat.items()):
        cur_p1 = sum(x.get("pass@1", 0.0) for x in items) / len(items)
        if base_map:
            b_vals = [base_map[x["id"]].get("pass@1", 0.0) for x in items if x["id"] in base_map]
            b_p1 = sum(b_vals) / max(1, len(b_vals))
            delta_s = f"{(cur_p1 - b_p1):+.1%}"
            base_s = f"{b_p1:.1%}"
        else:
            base_s, delta_s = "—", "—"
        table_lines.append(f"| `{cat}` | {len(items)} | {base_s} | **{cur_p1:.1%}** | {delta_s} |")

    ov_base_s = f"{base_overall:.1%}" if base_overall is not None else "—"
    ov_delta_s = f"{(overall - base_overall):+.1%}" if base_overall is not None else "—"
    table_lines.append(f"| **Overall** | **{len(eval_rows)}** | {ov_base_s} | **{overall:.1%}** | **{ov_delta_s}** |")

    audit_md = ""
    if audit_stats:
        audit_md = (
            "\n## Benchmark Decontamination Audit\n"
            f"- Verified training traces kept: **{audit_stats.get('clean_kept', 0)}**\n"
            f"- 12-gram benchmark leaks stripped (`threshold=0.50`): **{audit_stats.get('contaminated_removed', 0)}**\n"
            f"- Near-duplicates pruned: **{audit_stats.get('duplicates_removed', 0)}**\n"
        )

    readme = f"""---
language:
- en
license: apache-2.0
base_model: {base_model}
tags:
- code
- rlvr
- grpo
- dpo
- swe-agent
- neutron-ai
- hadron
---

# {model_name}

**{model_name}** is an execution-verified polyglot coding & repository-patching model trained by **NeutronAI** on top of `{base_model}` using dense per-assertion **RLVR (Dr.GRPO)** + **Self-Repair DPO** across Python, JavaScript, TypeScript, C++23, SQL, algorithmic speed gates, and multi-file `SEARCH/REPLACE` repository edits.

## Model Family Overview
- **Hadron (`Hadron-7B` / `Hadron-32B`)**: Flagship reasoning, debugging, and multi-file repository engineering models.
- **Quark (`Quark-1.5B`)**: Ultra-low-latency `<|file_sep|>` Cross-File Repo-FIM Tab model and speculative cascade draft model.

## Evaluation Results

{chr(10).join(table_lines)}
{audit_md}
## Quickstart (Ollama Local — Runs on CPU / Consumer GPUs)

```bash
ollama create {model_name.lower()} -f Modelfile
ollama run {model_name.lower()}
```

## Recommended System Prompts

- **Code Synthesis:** `Think step-by-step inside <think>...</think>, then output only valid <lang> code inside a fenced block.`
- **Multi-File Repository Editing:** `Think step-by-step inside <think>...</think>, then output exact ### FILE: <path> and <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE patch blocks.`
"""

    modelfile = f"""FROM {gguf_path}
PARAMETER temperature 0.2
PARAMETER num_ctx 8192
SYSTEM \"\"\"Think step-by-step inside <think>...</think>, then output clean, execution-verified code or exact ### FILE: <path> SEARCH/REPLACE blocks.\"\"\"
"""
    return readme, modelfile


def run_self_check():
    sample_eval = [
        {"id": "t1", "category": "repo", "pass@1": 1.0},
        {"id": "t2", "category": "cpp", "pass@1": 1.0},
    ]
    sample_base = [
        {"id": "t1", "category": "repo", "pass@1": 0.5},
        {"id": "t2", "category": "cpp", "pass@1": 0.5},
    ]
    md, mf = build_release_bundle(
        "Hadron-7B-v0.9",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "./hadron-7b-v0.9-q4_k_m.gguf",
        sample_eval,
        sample_base,
        {"clean_kept": 420, "contaminated_removed": 3, "duplicates_removed": 12},
    )
    assert "+50.0%" in md and "base_model: Qwen/Qwen2.5-Coder-7B-Instruct" in md and "NeutronAI" in md, md
    assert "FROM ./hadron-7b-v0.9-q4_k_m.gguf" in mf, mf
    print("release.py (NeutronAI) self-check passed: generated Hadron-7B Hugging Face Card + Leaderboard Delta + Modelfile.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Hadron-7B-v0.9")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--gguf", default="./hadron-7b-v0.9-gguf/unsloth.Q4_K_M.gguf")
    ap.add_argument("--eval-report", default="eval_report.jsonl")
    ap.add_argument("--base-report", default="")
    ap.add_argument("--out-dir", default="release_artifacts")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        run_self_check()
    else:
        with open(args.eval_report, encoding="utf-8") as f:
            eval_rows = [json.loads(l) for l in f if l.strip()]
        base_rows = None
        if args.base_report and os.path.exists(args.base_report):
            with open(args.base_report, encoding="utf-8") as fb:
                base_rows = [json.loads(l) for l in fb if l.strip()]
        md, mf = build_release_bundle(args.model_name, args.base_model, args.gguf, eval_rows, base_rows)
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(md)
        with open(os.path.join(args.out_dir, "Modelfile"), "w", encoding="utf-8") as f:
            f.write(mf)
        print(f"Wrote Hugging Face README.md + Ollama Modelfile -> {args.out_dir}/")
