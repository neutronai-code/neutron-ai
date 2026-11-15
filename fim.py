"""NeutronAI Quark Fill-in-the-Middle Engine (`fim.py`) — Execution-Verified Cross-File Repo-FIM Data + Evaluator.

Why every coding AI unicorn (Cursor Tab, Windsurf Supercomplete, Qwen-Coder, Codestral, DeepSeek)
trains heavily on Fill-in-the-Middle (`<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>`):
  - Chat answers 20% of developer work; **inline cursor autocomplete is 80% of daily keystrokes**.
  - Most labs evaluate FIM with fuzzy string BLEU/exact-match—which penalizes valid equivalent code!
  - NeutronAI trains & evaluates **Quark** FIM completions **by splicing `prefix + middle + suffix` and running
    the real unit tests in `sandbox.py`** (Python, JS, TypeScript, C++23, and `<|file_sep|>` multi-file repos)!

What `fim.py` does ($0, zero external dependencies):
  1. `mine`: Takes any verified source file + unit test and carves AST-aligned function-body, return-expression,
     and multi-line block spans into `(prefix, suffix, oracle_middle, test)` FIM training & RLVR tasks!
  2. `verify_fim`: Splices model middle completion into `prefix + middle + suffix` and verifies via `sandbox.py`
     (plus exposes `fim_grpo_reward_func` so `train.py` can RL-train ultra-fast `Quark-1.5B` Tab models!).
  3. `eval`: Benchmarks any FIM or chat endpoint on execution-verified FIM `pass@1`.

Usage:
  uv run fim.py --self-check
  uv run fim.py mine --out fim_tasks.jsonl
"""
import argparse, ast, json, re, urllib.request
from repomap import extract_file_skeleton, graph_neighbors
from sandbox import CODE_BLOCK, strip_think, verify

FIM_PRE = "<|fim_prefix|>"
FIM_SUF = "<|fim_suffix|>"
FIM_MID = "<|fim_middle|>"
FIM_SEP = "<|file_sep|>"


def extract_fim_middle(raw_reply: str) -> str:
    """Clean chat fences if queried via `/v1/chat/completions`, while preserving exact leading indentation!"""
    no_think = re.sub(r"<think>.*?</think>", "", raw_reply, flags=re.DOTALL | re.IGNORECASE)
    m = CODE_BLOCK.findall(no_think)
    if m:
        return m[-1].rstrip("\r\n")
    return no_think.strip("\r\n")


def format_qwen_fim_prompt(
    prefix: str,
    suffix: str,
    repo_files: dict[str, str] | None = None,
    target_path: str = "",
) -> str:
    """Standard Qwen2.5-Coder Repo-FIM layout (`<|file_sep|>...` cross-file skeletons + `<|fim_prefix|>...`)."""
    ctx_blocks = []
    if repo_files and target_path:
        for nb in graph_neighbors(repo_files, target_path)[:4]:
            skel = extract_file_skeleton(nb, repo_files[nb]) or repo_files[nb][:800]
            ctx_blocks.append(f"{FIM_SEP}{nb}\n{skel}\n")
        ctx_blocks.append(f"{FIM_SEP}{target_path}\n")
    return "".join(ctx_blocks) + f"{FIM_PRE}{prefix}{FIM_SUF}{suffix}{FIM_MID}"


def verify_fim(
    middle_text: str,
    prefix: str,
    suffix: str,
    test_code: str,
    lang: str = "python",
    repo_files: dict[str, str] | None = None,
    target_path: str = "",
    timeout: float = 3.5,
) -> tuple[float, dict]:
    """Splice `prefix + middle + suffix` (into single file OR multi-file `repo_files[target_path]`) and execute tests."""
    mid = extract_fim_middle(middle_text)
    if not mid.strip():
        return -0.5, {"status": "empty_middle", "passed": 0, "total": 0}
    assembled = f"{prefix}{mid}{suffix}"
    if repo_files and target_path:
        merged_files = {**repo_files, target_path: assembled}
        noop_patch = f"### FILE: {target_path}\n<<<<<<< SEARCH\n{assembled}\n=======\n{assembled}\n>>>>>>> REPLACE"
        return verify(noop_patch, test_code, files=merged_files, lang=lang, timeout=timeout)
    fenced = f"```{lang}\n{assembled}\n```"
    return verify(fenced, test_code, lang=lang, timeout=timeout)


def fim_grpo_reward_func(completions, prefix, suffix, test, lang=None, repo_files=None, target_path=None, **kwargs) -> list[float]:
    """TRL GRPOTrainer reward hook for single-file AND cross-file Repo-FIM autocomplete RLVR!"""
    n = len(completions)
    langs = list(lang) if lang is not None else ["python"] * n
    rfiles = [
        (json.loads(rf) if isinstance(rf, str) and rf else rf)
        for rf in (list(repo_files) if repo_files is not None else [None] * n)
    ]
    tpaths = list(target_path) if target_path is not None else [""] * n
    scores = []
    for comp, pre, suf, tst, lg, rf, tp in zip(completions, prefix, suffix, test, langs, rfiles, tpaths):
        txt = comp[0]["content"] if isinstance(comp, list) else comp
        try:
            s, _ = verify_fim(txt, pre, suf, tst, lang=lg, repo_files=rf, target_path=tp)
        except Exception as e:  # A verifier bug must never kill a paid training run; score it 0 and keep going.
            print(f"[fim] verifier crashed: {type(e).__name__}: {e}", flush=True)
            s = 0.0
        scores.append(s)
    return scores


def mine_fim_tasks_from_python(task_id: str, clean_code: str, test_code: str, max_spans: int = 3) -> list[dict]:
    """Carve AST-aligned `Return`, `Assign`, and loop/branch bodies into verified FIM tasks."""
    lines = clean_code.splitlines(keepends=True)
    tree = ast.parse(clean_code)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Return, ast.Assign, ast.AugAssign, ast.For, ast.While)) and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            lo, hi = node.lineno - 1, node.end_lineno
            if 0 <= lo < hi <= len(lines) and (hi - lo) <= 6:
                spans.append((lo, hi))

    # Deduplicate & prefer inner meaty spans
    seen, tasks = set(), []
    for lo, hi in spans:
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        prefix = "".join(lines[:lo])
        oracle_mid = "".join(lines[lo:hi]).rstrip("\r\n")
        suffix = "\n" + "".join(lines[hi:])
        if not oracle_mid.strip():
            continue
        # Confirm spliced oracle passes 100% AND blank hole fails!
        ok_score, _ = verify_fim(oracle_mid, prefix, suffix, test_code)
        hole_score, _ = verify_fim("    pass", prefix, suffix, test_code)
        if ok_score == 1.0 and hole_score < 1.0:
            tasks.append({
                "id": f"fim_{task_id}_L{lo+1}",
                "category": "fim",
                "lang": "python",
                "prefix": prefix,
                "suffix": suffix,
                "fim_prompt": format_qwen_fim_prompt(prefix, suffix),
                "oracle_middle": oracle_mid,
                "test": test_code,
            })
            if len(tasks) >= max_spans:
                break
    return tasks


def query_fim_via_chat(url: str, model: str, prefix: str, suffix: str, lang: str, api_key: str) -> str:
    payload = json.dumps({
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Complete ONLY the missing `{lang}` code between `<PREFIX>` and `<SUFFIX>` "
                    "inside a fenced code block, preserving exact indentation. Do not repeat prefix or suffix."
                ),
            },
            {"role": "user", "content": f"<PREFIX>\n{prefix}</PREFIX>\n<SUFFIX>\n{suffix}</SUFFIX>"},
        ],
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]


def run_self_check():
    from eval import MOCK_SOLUTIONS

    with open("tasks.jsonl", encoding="utf-8") as f:
        base_tasks = [json.loads(l) for l in f if l.strip()]

    # Mine FIM spans from verified Python solutions across tasks.jsonl
    mined_all = []
    for t in base_tasks[:5]:
        sol = re.sub(r"^```\w*\n|```$", "", MOCK_SOLUTIONS[t["id"]].strip())
        mined_all.extend(mine_fim_tasks_from_python(t["id"], sol, t["test"], max_spans=2))

    assert len(mined_all) >= 5, f"Expected >=5 mined FIM tasks, got {len(mined_all)}"
    rewards = fim_grpo_reward_func(
        [[{"content": f"```python\n{m['oracle_middle']}\n```"}] for m in mined_all],
        [m["prefix"] for m in mined_all],
        [m["suffix"] for m in mined_all],
        [m["test"] for m in mined_all],
    )
    assert all(r == 1.0 for r in rewards), rewards

    # Also verify polyglot TypeScript FIM splicing!
    ts_r, _ = verify_fim(
        "```ts\n  return items.filter(x => x > minVal);\n```",
        "function above(items: number[], minVal: number): number[] {\n",
        "\n}\n",
        "assert.deepEqual(above([1, 5, 9], 4), [5, 9]);",
        lang="typescript",
    )
    assert ts_r == 1.0, ts_r

    # Verify Cross-File Repo-FIM (`<|file_sep|>` neighbor skeleton injection + multi-file sandbox splice)!
    repo_fls = {
        "billing/__init__.py": "",
        "billing/rates.py": "def tax_multiplier(region: str) -> float:\n    return 1.08 if region == 'NY' else 1.0\n",
        "billing/checkout.py": "from billing.rates import tax_multiplier\n\ndef total_with_tax(subtotal: float, region: str) -> float:\n    return round(subtotal * tax_multiplier(region), 2)\n",
    }
    pre = "from billing.rates import tax_multiplier\n\ndef total_with_tax(subtotal: float, region: str) -> float:\n"
    suf = "\n"
    rfim_prompt = format_qwen_fim_prompt(pre, suf, repo_files=repo_fls, target_path="billing/checkout.py")
    assert "<|file_sep|>billing/rates.py" in rfim_prompt and "def tax_multiplier(region: str) -> float:" in rfim_prompt, rfim_prompt
    repo_fim_r, _ = verify_fim(
        "```python\n    return round(subtotal * tax_multiplier(region), 2)\n```",
        pre,
        suf,
        "from billing.checkout import total_with_tax\nassert total_with_tax(100.0, 'NY') == 108.0",
        repo_files=repo_fls,
        target_path="billing/checkout.py",
    )
    assert repo_fim_r == 1.0, repo_fim_r
    print(f"fim.py self-check passed: mined {len(mined_all)} AST FIM spans + verified Cross-File `<|file_sep|>` Repo-FIM & Py/TS splice.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", default="mine", choices=["mine", "eval"])
    ap.add_argument("--out", default="fim_tasks.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="quark-1.5b")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        run_self_check()
    elif args.action == "mine":
        from eval import MOCK_SOLUTIONS

        with open("tasks.jsonl", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        out_tasks = []
        for t in rows:
            if (t.get("lang") or "python") == "python" and not t.get("files") and t["id"] in MOCK_SOLUTIONS:
                clean = re.sub(r"^```\w*\n|```$", "", MOCK_SOLUTIONS[t["id"]].strip())
                out_tasks.extend(mine_fim_tasks_from_python(t["id"], clean, t["test"], max_spans=2))
        with open(args.out, "w", encoding="utf-8") as fo:
            for item in out_tasks:
                fo.write(json.dumps(item) + "\n")
        print(f"Mined {len(out_tasks)} execution-verified FIM autocomplete tasks -> {args.out}")
    else:
        with open(args.out, encoding="utf-8") as f:
            fim_rows = [json.loads(l) for l in f if l.strip()]
        wins = 0
        for r in fim_rows:
            reply = query_fim_via_chat(args.url, args.model, r["prefix"], r["suffix"], r.get("lang", "python"), args.api_key)
            sc, info = verify_fim(reply, r["prefix"], r["suffix"], r["test"], r.get("lang", "python"))
            wins += int(sc == 1.0)
            print(f"[{r['id']:30s}] {'PASS' if sc == 1.0 else 'FAIL'} ({info['status']})")
        print(f"\nFIM execution pass@1: {wins}/{len(fim_rows)} ({wins/max(1,len(fim_rows)):.1%})")
