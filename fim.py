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
import argparse, ast, collections, concurrent.futures, json, random, re, urllib.error, urllib.request
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
    fence = {"python_stdio": "python", "stdio": "python"}.get(lang, lang)  # CODE_BLOCK only knows real language tags
    fenced = f"```{fence}\n{assembled}\n```"
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


def python_spans(code: str, hard: bool = False) -> list[tuple[int, int]]:
    """AST-aligned hole spans (0-based [lo, hi) line ranges) in Python code, in AST order.
    Normal: 1-6 line Return/Assign/loop/branch statements. hard=True: 3-15 lines, adding with/try blocks and whole
    function bodies (keeping the docstring, which tells the model what the body should do)."""
    lines = code.splitlines(keepends=True)
    tree = ast.parse(code)
    min_n, max_n = (3, 15) if hard else (1, 6)
    kinds = (ast.Return, ast.Assign, ast.AugAssign, ast.For, ast.While, ast.If)
    if hard:
        kinds += (ast.With, ast.Try)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, kinds) and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            spans.append((node.lineno - 1, node.end_lineno))
        if hard and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            body = node.body
            if len(body) > 1 and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                body = body[1:]
            spans.append((body[0].lineno - 1, node.end_lineno))
    return [(lo, hi) for lo, hi in spans if 0 <= lo < hi <= len(lines) and min_n <= hi - lo <= max_n]


def mine_fim_tasks_from_python(task_id: str, clean_code: str, test_code: str, max_spans: int = 3,
                               lang: str = "python", shuffle_seed: str = "", hard: bool = False) -> list[dict]:
    """Carve AST-aligned `Return`, `Assign`, and loop/branch bodies into verified FIM tasks.
    hard=True: 3-15 line holes instead (whole loops/branches/with/try blocks and whole function bodies)."""
    lines = clean_code.splitlines(keepends=True)
    spans = python_spans(clean_code, hard)
    if shuffle_seed:  # vary which spans get picked (deterministically per case) instead of always the first ones
        random.Random(shuffle_seed).shuffle(spans)
    return _verified_fim_tasks(task_id, lines, spans, test_code, lang, "    pass", max_spans, "_H" if hard else "")


def brace_spans(lines: list[str], hard: bool = False) -> list[tuple[int, int]]:
    """Brace languages (JS/TS/C++): 1-4 consecutive indented statement lines whose braces balance.
    hard=True: 3-12 line spans that may contain blank lines and closing braces, i.e. whole nested blocks."""
    closers = ("{", "}", "};", "});", ")")
    comment = ("//", "/*", "*", "#")
    if hard:
        ok_line = [not ln.strip() or ln[:1] in (" ", "	") for ln in lines]
        can_start = [bool(ln.strip()) and ln[:1] in (" ", "	") and not ln.strip().startswith(("}", ")") + comment)
                     for ln in lines]
        sizes = range(1, 13)
    else:
        ok_line = [bool(ln.strip()) and ln[:1] in (" ", "	") and ln.strip() not in closers
                   and not ln.strip().startswith(comment) for ln in lines]
        can_start = ok_line
        sizes = range(1, 5)
    spans = []
    for lo in range(len(lines)):
        if not can_start[lo]:
            continue
        for size in sizes:
            hi = lo + size
            if hi > len(lines) or not ok_line[hi - 1]:
                break
            if hard and (size < 3 or not lines[hi - 1].strip()):
                continue
            chunk = "".join(lines[lo:hi])
            if all(chunk.count(a) == chunk.count(b) for a, b in (("{}", "()", "[]") if hard else ("{}", "()"))):
                spans.append((lo, hi))
    return spans


def mine_fim_tasks_from_lines(task_id: str, code: str, test_code: str, lang: str, max_spans: int = 2,
                              shuffle_seed: str = "", hard: bool = False) -> list[dict]:
    """Brace languages (JS/TS/C++): carve 1-4 consecutive indented statement lines whose braces balance.
    hard=True: 3-12 line holes that may contain blank lines and closing braces, i.e. whole nested blocks."""
    lines = code.splitlines(keepends=True)
    spans = brace_spans(lines, hard)
    random.Random(shuffle_seed or task_id).shuffle(spans)
    return _verified_fim_tasks(task_id, lines, spans, test_code, lang, "/* TODO */", max_spans, "_H" if hard else "")


def _verified_fim_tasks(task_id: str, lines: list[str], spans: list[tuple[int, int]], test_code: str, lang: str,
                        hole: str, max_spans: int, id_tag: str = "") -> list[dict]:
    """Keep spans where the spliced oracle passes AND a blank hole fails (so the middle really matters)."""
    seen, tasks, original_ok = set(), [], None
    for lo, hi in spans:
        if (lo, hi) in seen or any(lo < h and l < hi for l, h in seen):  # no overlapping holes per case
            continue
        prefix = "".join(lines[:lo])
        oracle_mid = "".join(lines[lo:hi]).rstrip("\r\n")
        suffix = "\n" + "".join(lines[hi:])
        if not oracle_mid.strip():
            continue
        if original_ok is None:  # splicing the oracle back rebuilds the original file, so check that only once
            original_ok = verify_fim(oracle_mid, prefix, suffix, test_code, lang=lang)[0] == 1.0
        if not original_ok:
            return []
        hole_score, _ = verify_fim(hole, prefix, suffix, test_code, lang=lang)
        if hole_score >= 1.0:
            continue
        seen.add((lo, hi))
        tasks.append({
            "id": f"fim_{task_id}_L{lo+1}{id_tag}",
            "category": "fim",
            "lang": lang,
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


# vLLM (like OpenAI) accepts at most 4 stop strings. These are special tokens, so the request also sets
# skip_special_tokens=False: otherwise they never appear in the text and the stop strings could not match.
FIM_STOPS = ["<|endoftext|>", "<|fim_pad|>", "<|file_sep|>", "<|fim_prefix|>"]


MELLUM_STOPS = ["<|endoftext|>", "<filename>", "<fim_prefix>", "<fim_suffix>"]
GRANITE_STOPS = ["<|end_of_text|>", "<|filename|>", "<|fim_prefix|>", "<|fim_pad|>"]

_FIM_STYLE: dict[str, str] = {}


def fim_style(model: str) -> str:
    """Which FIM prompt format a model uses, from its tokenizer on the Hub (so the name doesn't matter and a fine-tune
    inherits its base's format): "starcoder" (JetBrains Mellum: `<fim_prefix>`, `<filename>`, suffix first), "granite"
    (IBM Granite: Qwen's FIM tokens but `<|filename|>` for repo files), or "qwen". Cached per model."""
    if model not in _FIM_STYLE:
        try:
            from huggingface_hub import hf_hub_download

            # tokenizer_config.json first (small), then tokenizer.json: transformers 5 saves fine-tunes' configs without
            # their special-token list, so a Mellum fine-tune's `<fim_prefix>` only shows in tokenizer.json.
            style = "qwen"
            for name in ("tokenizer_config.json", "tokenizer.json"):
                with open(hf_hub_download(model, name), encoding="utf-8") as f:
                    tok = f.read()
                if '"<fim_prefix>"' in tok or '"<|filename|>"' in tok:
                    style = "starcoder" if '"<fim_prefix>"' in tok else "granite"
                    break
            _FIM_STYLE[model] = style
        except Exception:
            _FIM_STYLE[model] = "starcoder" if "mellum" in model.lower() else "qwen"
    return _FIM_STYLE[model]


def is_mellum(model: str) -> bool:
    return fim_style(model) == "starcoder"


def to_mellum_prompt(prompt: str) -> str:
    """A Qwen-format FIM prompt (`[<|file_sep|>path\n...]<|fim_prefix|>P<|fim_suffix|>S<|fim_middle|>`) in JetBrains
    Mellum's format: `[<filename>path\n...]<fim_suffix>S<fim_prefix>P<fim_middle>` (suffix first, as on its card)."""
    i = prompt.index(FIM_PRE)
    ctx, rest = prompt[:i], prompt[i + len(FIM_PRE):]
    pre, rest = rest.split(FIM_SUF, 1)
    suf = rest[:rest.rindex(FIM_MID)]
    return ctx.replace(FIM_SEP, "<filename>") + "<fim_suffix>" + suf + "<fim_prefix>" + pre + "<fim_middle>"


def to_granite_prompt(prompt: str) -> str:
    """Granite uses Qwen's FIM tokens in the same order, but `<|filename|>` instead of `<|file_sep|>`."""
    return prompt.replace(FIM_SEP, "<|filename|>")


def convert_fim_prompt(prompt: str, style: str) -> str:
    return {"starcoder": to_mellum_prompt, "granite": to_granite_prompt}.get(style, lambda p: p)(prompt)


def query_fim_samples(url: str, model: str, fim_prompt: str, api_key: str = "", max_tokens: int = 256,
                      temperature: float = 0.0, n: int = 1) -> list[str]:
    """Raw FIM for base models via /v1/completions: the prompt is `<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>`
    (converted for models with another FIM format: see fim_style)."""
    style = fim_style(model)
    fim_prompt = convert_fim_prompt(fim_prompt, style)
    stops = {"starcoder": MELLUM_STOPS, "granite": GRANITE_STOPS}.get(style, FIM_STOPS)
    payload = json.dumps({"model": model, "prompt": fim_prompt, "max_tokens": max_tokens, "temperature": temperature,
                          "n": n, "stop": stops, "skip_special_tokens": False}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return [c["text"] for c in json.loads(resp.read().decode("utf-8"))["choices"]]
    except urllib.error.HTTPError as e:  # surface the server's reason (e.g. a 400 message) instead of just the code
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None


def query_fim_completion(url: str, model: str, fim_prompt: str, api_key: str = "", max_tokens: int = 256) -> str:
    return query_fim_samples(url, model, fim_prompt, api_key, max_tokens)[0]


def _clean_oracle(code: str) -> str:
    m = CODE_BLOCK.findall(code or "")
    return (m[-1] if m else code or "").strip("\n")


def _mine_case(t: dict, max_spans: int, hard: bool = False) -> list[dict]:
    lang, code = t.get("lang", "python"), _clean_oracle(t.get("oracle_solution", ""))
    if not code or t.get("files") or t.get("kind") == "repo_python" or lang == "sql":
        return []
    seed = t["id"] + (":hard" if hard else "")
    try:
        if lang in ("python", "python_stdio"):
            return mine_fim_tasks_from_python(t["id"], code, t["test"], max_spans, lang=lang, shuffle_seed=seed, hard=hard)
        if lang in ("javascript", "typescript", "cpp"):
            return mine_fim_tasks_from_lines(t["id"], code, t["test"], lang, max_spans, shuffle_seed=seed, hard=hard)
    except (SyntaxError, ValueError) as e:
        print(f"  skip {t['id']}: {type(e).__name__}", flush=True)
    return []


def filter_fim(data: str, out: str, url: str, model: str, samples: int = 4, temperature: float = 0.8,
               workers: int = 32, limit: int = 0) -> int:
    """Difficulty filter: the model answers every hole `samples` times. Holes it always fills correctly teach RL
    nothing and are dropped; the rest get `baseline_pass_rate` and are written most-informative first (~50% pass
    rate, then never-solved), which train.py keeps as the training order."""
    with open(data, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if limit:
        rows = rows[:limit]

    def one(r):
        try:
            replies = query_fim_samples(url, model, r["fim_prompt"], temperature=temperature, n=samples)
        except Exception as e:  # noqa: BLE001 - server trouble: leave the hole out rather than guess its rate
            return None, f"{type(e).__name__}: {str(e)[:200]}"
        lang = r.get("lang", "python")
        wins = sum(verify_fim(c, r["prefix"], r["suffix"], r["test"], lang)[0] == 1.0 for c in replies)
        return wins / len(replies), ""

    done, kept, errors = 0, [], collections.Counter()
    table = collections.defaultdict(collections.Counter)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for r, (rate, err) in zip(rows, pool.map(one, rows)):
            done += 1
            if rate is None:
                errors[err] += 1
            else:
                bucket = "always" if rate == 1.0 else "never" if rate == 0.0 else "sometimes"
                table[r.get("lang", "python")][bucket] += 1
                if rate < 1.0:
                    kept.append({**r, "baseline_pass_rate": round(rate, 3)})
            if done % 500 == 0:
                print(f"  {done}/{len(rows)} holes scored, {len(kept)} kept", flush=True)
    for msg, n in errors.most_common(3):
        print(f"  {n} requests failed: {msg}", flush=True)
    random.Random(42).shuffle(kept)  # mix languages within each difficulty level
    kept.sort(key=lambda r: abs(r["baseline_pass_rate"] - 0.5))
    with open(out, "w", encoding="utf-8") as fo:
        for r in kept:
            fo.write(json.dumps(r) + "\n")
    print(f"\n{'lang':14s} {'always':>7s} {'sometimes':>10s} {'never':>6s}")
    for lang, c in sorted(table.items()):
        print(f"{lang:14s} {c['always']:7d} {c['sometimes']:10d} {c['never']:6d}")
    print(f"Kept {len(kept)} of {len(rows)} holes (dropped the always-solved ones) -> {out}")
    return len(kept)


def mine_from_cases(source: str, holdout_path: str, out: str, holdout_out: str, max_spans: int, workers: int, limit: int,
                    hard: bool = False):
    with open(source, encoding="utf-8") as f:
        cases = [json.loads(l) for l in f if l.strip()]
    if limit:
        cases = cases[:limit]
    held = set()
    if holdout_path:
        with open(holdout_path, encoding="utf-8") as f:
            held = {json.loads(l)["id"] for l in f if l.strip()}
    train, test, done = [], [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for t, mined in zip(cases, pool.map(lambda c: _mine_case(c, max_spans, hard), cases)):
            for m in mined:
                m["source_case"] = t["id"]
            (test if t["id"] in held else train).extend(mined)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(cases)} cases -> {len(train)} train + {len(test)} held-out FIM tasks", flush=True)
    for path, rows in ((out, train), (holdout_out, test)):
        with open(path, "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r) + "\n")
    by_lang = collections.Counter(r["lang"] for r in train + test)
    print(f"Mined {len(train)} training FIM tasks -> {out} and {len(test)} held-out -> {holdout_out} "
          f"({', '.join(f'{k} {v}' for k, v in sorted(by_lang.items()))})")


def eval_fim(data: str, url: str, model: str, api_key: str, mode: str, workers: int, limit: int, report: str):
    with open(data, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if limit:
        rows = rows[:limit]

    def one(r):
        lang = r.get("lang", "python")
        try:
            reply = (query_fim_completion(url, model, r["fim_prompt"], api_key) if mode == "completions"
                     else query_fim_via_chat(url, model, r["prefix"], r["suffix"], lang, api_key))
            sc, info = verify_fim(reply, r["prefix"], r["suffix"], r["test"], lang)
        except Exception as e:  # noqa: BLE001 - count server/verifier trouble as a miss, keep going
            reply, sc, info = "", 0.0, {"status": f"error: {type(e).__name__}: {str(e)[:300]}"}
        return {"id": r["id"], "lang": lang, "pass": sc == 1.0, "status": info.get("status", ""), "completion": reply}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, rows))
    errors = collections.Counter(r["status"] for r in results if r["status"].startswith("error"))
    for msg, n in errors.most_common(3):
        print(f"  {n} requests failed: {msg}", flush=True)
    by_lang = collections.defaultdict(list)
    for res in results:
        by_lang[res["lang"]].append(res["pass"])
    print(f"\nFIM pass@1 for {model} ({mode}):")
    for lang, ok in sorted(by_lang.items()):
        print(f"  {lang:14s} {sum(ok):4d}/{len(ok):<4d} {sum(ok) / len(ok):.1%}")
    total = sum(r["pass"] for r in results)
    print(f"  {'overall':14s} {total:4d}/{len(results):<4d} {total / max(1, len(results)):.1%}")
    if report:
        with open(report, "w", encoding="utf-8") as fo:
            for res in results:
                fo.write(json.dumps(res) + "\n")
    return total / max(1, len(results))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", default="mine", choices=["mine", "eval", "filter"])
    ap.add_argument("--hard", action="store_true", help="mine: longer holes (3-15 lines: whole blocks / function bodies)")
    ap.add_argument("--samples", type=int, default=4, help="filter: answers per hole")
    ap.add_argument("--temperature", type=float, default=0.8, help="filter: sampling temperature")
    ap.add_argument("--source", default="", help="mine: verified cases file (e.g. tasks_generated.jsonl); empty = legacy tasks.jsonl")
    ap.add_argument("--holdout", default="tasks_holdout.jsonl", help="mine: cases whose FIM tasks go to --holdout-out")
    ap.add_argument("--out", default="fim_tasks.jsonl")
    ap.add_argument("--holdout-out", default="fim_holdout.jsonl")
    ap.add_argument("--max-spans", type=int, default=2)
    ap.add_argument("--data", default="", help="eval: FIM tasks file (default --out)")
    ap.add_argument("--mode", choices=["chat", "completions"], default="chat",
                    help="eval: chat = instruct models; completions = raw <|fim_prefix|> prompt for base models")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="quark-1.5b")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default="")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        run_self_check()
    elif args.action == "mine" and args.source:
        mine_from_cases(args.source, args.holdout, args.out, args.holdout_out, args.max_spans, args.workers, args.limit,
                        hard=args.hard)
    elif args.action == "filter":
        filter_fim(args.data or args.out, args.report or "fim_filtered.jsonl", args.url, args.model, args.samples,
                   args.temperature, args.workers, args.limit)
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
        eval_fim(args.data or args.out, args.url, args.model, args.api_key, args.mode, args.workers, args.limit, args.report)
