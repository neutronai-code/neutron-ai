"""Build Hadron's next SFT set: Mellum's own verified reasoning + the 282 hard-task teacher traces.

The existing hadron-data answer files kept only final code, so first run molab_hadron_bench.py's
"Native SFT harvest" task set. That mode stores `reasoning_content` as a <think> block. This script re-verifies
each harvested answer in the real sandbox, then mixes it with sft_distill_mellum_clean.jsonl.

  uv run --with transformers --with jinja2 --with huggingface_hub python build_sft_mix.py
  uv run audit.py --train sft_hadron_mix.jsonl --eval tasks_holdout.jsonl --clean-out sft_hadron_mix_clean.jsonl

Optional: --native-file path/to/harvest.json (local file instead of downloading it), --limit 20 (debugging),
--max-tokens 3072 (drop any row that would be truncated by train.py --mode sft).
"""
import argparse
import collections
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repair import system_prompt_for, task_prompt
from sandbox import verify

BASE_TAG = "Mellum2.1-12B-A2.5B-Thinking-32k"
DEFAULT_NATIVE = f"gen/{BASE_TAG}/train/harvest-000.json"


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", default=DEFAULT_NATIVE, help="answer file inside hadron-data")
    ap.add_argument("--native-file", default="", help="use this local answer file instead of downloading")
    ap.add_argument("--dataset", default="neutronai-code/hadron-data")
    ap.add_argument("--teacher", default="sft_distill_mellum_clean.jsonl")
    ap.add_argument("--out", default="sft_hadron_mix.jsonl")
    ap.add_argument("--native-out", default="sft_native_mellum.jsonl")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--limit", type=int, default=0, help="verify at most N harvested tasks (0 = all)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("JetBrains/Mellum2.1-12B-A2.5B-Thinking")
    tasks = {t["id"]: t for t in read_jsonl(os.path.join(HERE, "tasks_generated.jsonl"))}
    held = {t["id"] for name in ("tasks_holdout.jsonl", "tasks.jsonl") for t in read_jsonl(os.path.join(HERE, name))}
    if args.native_file:
        with open(args.native_file, encoding="utf-8") as f:
            gen = json.load(f)
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(args.dataset, args.native, repo_type="dataset", token=os.environ.get("HF_TOKEN") or None,
                               local_dir=os.path.join(HERE, ".native_cache"))
        with open(path, encoding="utf-8") as f:
            gen = json.load(f)
    answers = gen.get("answers", {})
    if not gen.get("complete"):
        print(f"WARNING: {args.native} is incomplete; using the {len(answers)} answers present", flush=True)

    def rendered_tokens(messages):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    native, skipped = [], collections.Counter()
    items = list(answers.items())[: args.limit or None]
    for i, (tid, candidates) in enumerate(items, 1):
        task = tasks.get(tid)
        if task is None or tid in held:
            skipped["benchmark_or_unknown"] += 1
            continue
        best = None
        for content, _finish, _tokens in sorted(candidates, key=lambda a: a[2] or 2**31):
            if "</think>" not in content:
                skipped["no_reasoning"] += 1
                continue
            score, _info = verify(content, task["test"], task.get("starter_code", ""), task.get("files"),
                                  task.get("lang", "python"), perf_budget_ms=float(task.get("perf_budget_ms", 0.0) or 0.0))
            if score == 1.0:
                best = content.strip()
                break
        if best is None:
            skipped["not_verified"] += 1
            continue
        messages = [{"role": "system", "content": system_prompt_for(task)},
                    {"role": "user", "content": task_prompt(task)},
                    {"role": "assistant", "content": best}]
        n_tokens = rendered_tokens(messages)
        if n_tokens > args.max_tokens:
            skipped["too_long"] += 1
            continue
        native.append({"id": tid, "teacher": BASE_TAG, "messages": messages})
        if i % 25 == 0:
            print(f"  [{i}/{len(items)}] kept {len(native)}", flush=True)

    teacher = read_jsonl(os.path.join(HERE, args.teacher))
    native_ids = {r["id"] for r in native}
    teacher = [r for r in teacher if r["id"] not in native_ids and r["id"] not in held]
    teacher = [r for r in teacher if rendered_tokens(r["messages"]) <= args.max_tokens]
    mixed = native + teacher
    random.Random(0).shuffle(mixed)

    for path, rows in ((args.native_out, native), (args.out, mixed)):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    token_counts = sorted(rendered_tokens(r["messages"]) for r in mixed)
    print(f"\nNative verified traces: {len(native)} -> {args.native_out}")
    print(f"Teacher traces: {len(teacher)} | mixed: {len(mixed)} -> {args.out}")
    print(f"Native/teacher ratio: {len(native) / max(1, len(teacher)):.2f}:1")
    print(f"Skipped native answers: {dict(skipped)}")
    print(f"Rendered tokens: median {int(statistics.median(token_counts))}, p90 "
          f"{token_counts[int(0.9 * len(token_counts))]}, max {max(token_counts)}")
    print("Next: uv run audit.py --train sft_hadron_mix.jsonl --eval tasks_holdout.jsonl "
          "--clean-out sft_hadron_mix_clean.jsonl")


if __name__ == "__main__":
    main()
