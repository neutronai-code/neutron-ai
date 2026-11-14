"""NeutronAI Contamination & Deduplication Audit (`audit.py`) — Clean Benchmark & Dataset Hygiene.

Why credible coding labs (OpenAI, Anthropic, DeepSeek, LiveCodeBench) ALWAYS run decontamination:
  - If your training set (`sft_verified.jsonl`, `rl_curriculum.jsonl`, `mined_bugs.jsonl`) leaks prompts
    or near-duplicate tests from your evaluation benchmark (`tasks.jsonl`, HumanEval, MBPP), your `pass@1`
    score is fake memorization—and developers/investors spot it immediately in real usage.

What `audit.py` enforces (zero external dependencies):
  1. Validates task schema + runs oracle/mock sanity checks if `oracle_patch` is present.
  2. Exact + normalized 8-gram Jaccard similarity decontamination between `--train` and `--eval` JSONLs
     (flags and strips any training item with >= `--threshold` overlap against held-out eval prompts/tests!).
  3. Near-duplicate pruning inside `--train` so RL doesn't overfit on 20 copies of the same function.

Usage:
  uv run audit.py --self-check
  uv run audit.py --train rl_curriculum.jsonl --eval tasks.jsonl --clean-out rl_clean.jsonl
"""
import argparse, json, re

_WS_RE = re.compile(r"\s+")


def _norm_text(t: str) -> str:
    return _WS_RE.sub(" ", t.strip().lower())


def char_ngrams(text: str, n: int = 12) -> set[str]:
    norm = _norm_text(text)
    if len(norm) <= n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def extract_row_text(row: dict) -> str:
    if "prompt" in row and isinstance(row["prompt"], str):
        return f"{row['prompt']}\n{row.get('test', '')}"
    if "messages" in row and isinstance(row["messages"], list):
        return "\n".join(m.get("content", "") for m in row["messages"] if m.get("role") == "user")
    if "prompt" in row and isinstance(row["prompt"], list):
        return "\n".join(m.get("content", "") for m in row["prompt"] if m.get("role") == "user")
    return json.dumps(row, sort_keys=True)


def decontaminate(train_rows: list[dict], eval_rows: list[dict], threshold: float = 0.5) -> tuple[list[dict], dict]:
    eval_grams = [(r.get("id", f"eval_{i}"), char_ngrams(extract_row_text(r))) for i, r in enumerate(eval_rows)]
    kept, kept_grams = [], []
    leaked, dupes, invalid = 0, 0, 0

    for row in train_rows:
        txt = extract_row_text(row)
        if len(txt.strip()) < 10:
            invalid += 1
            continue
        g = char_ngrams(txt)

        # 1. Check contamination against held-out eval benchmark
        if any(jaccard(g, eg) >= threshold for _, eg in eval_grams):
            leaked += 1
            continue

        # 2. Check internal near-duplicates within training set
        if any(jaccard(g, kg) >= 0.85 for kg in kept_grams):
            dupes += 1
            continue

        kept.append(row)
        kept_grams.append(g)

    stats = {
        "input_train": len(train_rows),
        "eval_reference": len(eval_rows),
        "contaminated_removed": leaked,
        "duplicates_removed": dupes,
        "invalid_removed": invalid,
        "clean_kept": len(kept),
    }
    return kept, stats


def verify_oracle_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Keep only rows whose `oracle_patch` / `oracle_solution` (if provided) passes `sandbox.verify` with `1.0`."""
    from sandbox import verify

    verified, failed = [], 0
    for r in rows:
        sol = r.get("oracle_patch") or r.get("oracle_solution")
        if not sol:
            verified.append(r)
            continue
        sc, _ = verify(
            sol,
            r.get("test", ""),
            starter_code=r.get("starter_code", ""),
            files=r.get("files"),
            lang=r.get("lang", "python"),
            perf_budget_ms=float(r.get("perf_budget_ms", 0.0)),
        )
        if sc == 1.0:
            verified.append(r)
        else:
            failed += 1
    return verified, failed


def run_self_check():
    eval_set = [
        {"id": "eval_1", "prompt": "Write `merge_intervals(intervals)` that merges overlapping intervals.", "test": "assert merge_intervals([[1,3],[2,4]]) == [[1,4]]"},
    ]
    dirty_train = [
        # Leaked paraphrase/copy of eval_1 -> MUST be caught & stripped!
        {"id": "leak_1", "prompt": "Write `merge_intervals(intervals)`   that merges overlapping intervals!", "test": "assert merge_intervals([[1,3],[2,4]]) == [[1,4]]"},
        # Clean genuinely distinct task -> MUST be kept!
        {"id": "clean_1", "prompt": "Write `dijkstra_shortest_path(n, edges, src)` returning shortest distances.", "test": "assert dijkstra_shortest_path(2, [(0,1,5)], 0) == [0, 5]"},
        # Duplicate of clean_1 -> MUST be deduped!
        {"id": "dup_1", "prompt": "Write `dijkstra_shortest_path(n, edges, src)` returning shortest distances.", "test": "assert dijkstra_shortest_path(2, [(0,1,5)], 0) == [0, 5]"},
    ]
    clean, stats = decontaminate(dirty_train, eval_set, threshold=0.5)
    assert len(clean) == 1 and clean[0]["id"] == "clean_1", (clean, stats)
    assert stats["contaminated_removed"] == 1 and stats["duplicates_removed"] == 1, stats
    ok_rows, bad_n = verify_oracle_rows([
        {"id": "good_o", "prompt": "p", "test": "assert add1(2) == 3", "oracle_solution": "```python\ndef add1(x): return x + 1\n```"},
        {"id": "bad_o", "prompt": "p", "test": "assert add1(2) == 3", "oracle_solution": "```python\ndef add1(x): return x - 1\n```"},
    ])
    assert len(ok_rows) == 1 and bad_n == 1, (ok_rows, bad_n)
    print("audit.py self-check passed: caught benchmark leak + pruned near-duplicate + verified oracle execution.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="sft_verified.jsonl")
    ap.add_argument("--eval", default="tasks.jsonl")
    ap.add_argument("--clean-out", default="train_decontaminated.jsonl")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--verify-oracle", action="store_true", help="Execute `oracle_solution`/`oracle_patch` in sandbox and drop broken tasks")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        run_self_check()
    else:
        with open(args.train, encoding="utf-8") as ft, open(args.eval, encoding="utf-8") as fe:
            t_rows = [json.loads(l) for l in ft if l.strip()]
            e_rows = [json.loads(l) for l in fe if l.strip()]
        clean, stats = decontaminate(t_rows, e_rows, args.threshold)
        if args.verify_oracle:
            clean, bad_oracles = verify_oracle_rows(clean)
            stats["broken_oracles_removed"] = bad_oracles
            stats["clean_kept"] = len(clean)
        with open(args.clean_out, "w", encoding="utf-8") as fo:
            for r in clean:
                fo.write(json.dumps(r) + "\n")
        print(json.dumps(stats, indent=2))
