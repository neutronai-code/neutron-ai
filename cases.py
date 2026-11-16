"""NeutronAI Case Factory (`cases.py`) — turn a chat app's replies into verified training tasks.

No API needed: you paste a generated request into any strong model's chat app (one that may be used to create
training data), copy its reply, and `ingest` checks every case with the real sandbox before keeping it.

  1. uv run cases.py prompt --lang cpp --n 25          # prints a request (also copied to your clipboard)
  2. paste it into the app, then copy the app's whole reply
  3. uv run cases.py ingest --clipboard --source deepseek-v4.1-flash
     (or: uv run cases.py ingest reply.txt)

A case is kept only if:
  - its reference solution scores 1.0 in sandbox.verify (so the tests and solution agree),
  - its tests are strong enough (enough asserts / test cases; Python tests resist sham solutions via fuzz_tests;
    stdin tasks need varied outputs; repo tasks must actually fail BEFORE the fix),
  - it is not a near-duplicate of a case you already have.
Kept cases are appended to tasks_generated.jsonl; rejected ones go to cases_rejected.jsonl with the reason.
Languages: python, javascript, typescript, cpp, sql, python_stdio, repo_python.
"""
import argparse, ast, concurrent.futures, hashlib, json, os, random, re, sys

from audit import char_ngrams, extract_row_text, jaccard
from sandbox import _CPP_ASSERT_LINE, verify

TOPICS = [
    "interval scheduling", "rate limiting", "LRU / LFU caching", "graph shortest paths", "topological ordering",
    "union-find connectivity", "string parsing with escapes", "tokenizing a mini-language", "expression evaluation",
    "date and time arithmetic", "money and rounding", "inventory bookkeeping", "matrix manipulation",
    "sliding windows", "two pointers", "monotonic stacks", "heaps and priority scheduling", "binary search on answers",
    "dynamic programming on sequences", "DP on grids", "knapsack variants", "bit manipulation", "prefix sums",
    "segment / Fenwick trees", "tries and prefix search", "run-length and other encodings", "diffing two sequences",
    "version string comparison", "dependency resolution", "event sourcing / replaying logs", "text wrapping and alignment",
    "CSV / INI / query-string parsing", "URL and path normalization", "geometry on integer points", "simulation of a game",
    "state machines", "bank / ledger transfers", "leaderboards and ranking with ties", "calendar free-slot finding",
    "deduplication with fuzzy keys", "permissions and role checks", "retry / backoff policies", "pagination",
    "tree traversal and serialization", "recursive data flattening", "cycle detection", "scheduling with deadlines",
    "number theory (gcd, modular arithmetic)", "combinatorics counting mod 1e9+7", "string matching (KMP / Z / hashing)",
]

COMMON_RULES = """\
General rules (follow exactly):
- Return ONLY a JSON array with exactly {n} objects, inside ONE ```json code block. No text before or after it.
- Every object has these keys: "lang" (always "{lang}"), "topic", "prompt", "solution", "test"{extra_keys}.
- All code goes inside JSON strings (escape newlines as \\n and quotes as \\"). Do not use code fences inside strings.
- The reference "solution" MUST pass every test. Mentally run each test against it before writing it down.
- Difficulty mix: about 30% medium, 50% hard, 20% very hard (needs a non-obvious idea or careful edge cases).
- Avoid textbook classics (FizzBuzz, two-sum, reversing strings, plain Fibonacci). Make each case realistic and distinct.
- Tests must cover edge cases (empty input, single element, duplicates, boundaries, large values) and must NOT be passable
  by a hard-coded or constant answer. Never use randomness, the current time, the network, or files.
- Use these topics, one per case, in order: {topics}.
"""

LANG_RULES = {
    "python": """\
Language: Python 3 (standard library only).
- "prompt": describe the task and name the function in backticks with its full signature, e.g. `merge_ranges(ranges: list[list[int]]) -> list[list[int]]`.
- "solution": Python code defining that function (and helpers). No input(), no printing, no top-level test code.
- "test": Python code made of top-level `assert` statements (8-15 of them) that call the function directly (it is already in scope; do not import it). Helper variables are fine.""",
    "javascript": """\
Language: JavaScript (Node.js 22+, CommonJS).
- "prompt": describe the task and name the function in backticks with its parameters.
- "solution": plain JavaScript defining the function(s). No imports, no exports, no require of fs/net/child_process, no console output.
- "test": one statement per line. `assert` (node:assert/strict) is already available. Use assert.equal / assert.deepEqual / assert.throws, and put EACH assert call on ONE single line. 8-15 asserts. Setup lines like `const x = ...;` are fine.""",
    "typescript": """\
Language: TypeScript (run by Node.js type stripping).
- "prompt": describe the task and name the function in backticks with its typed signature.
- "solution": TypeScript defining the function(s), with type annotations. Do NOT use `enum`, `namespace`, or constructor parameter properties (`constructor(private x: number)`). No imports/exports, no console output.
- "test": one statement per line. `assert` (node:assert/strict) is already available. Use assert.equal / assert.deepEqual / assert.throws, EACH assert call on ONE single line. 8-15 asserts.""",
    "cpp": """\
Language: C++23.
- "prompt": describe the task and give the exact function / class signature in backticks.
- "solution": C++ functions/classes only. NO main(). These headers are already included: iostream vector string string_view algorithm unordered_map unordered_set map set array deque list tuple queue stack numeric cmath cstdint optional utility functional; #include anything else you need. No stdin, files, or printing.
- "test": the body of main(): setup statements plus `assert(expression);` lines, EACH assert on ONE single line, 8-15 asserts. No other output.""",
    "sql": """\
Language: SQLite SQL.
- "starter_code": SQLite statements that CREATE the tables and INSERT enough seed rows to make the query non-trivial (include ties, NULLs, and edge rows where relevant).
- "prompt": describe the schema and exactly what the query must return, including the exact column order and ORDER BY.
- "solution": ONE SQLite query (SELECT or WITH ... SELECT), no other statements.
- "test": Python `assert` statements (5-10) about `rows`, the list of tuples the query returned, e.g. `assert rows[0] == ('eng', 250)` and `assert len(rows) == 3`.""",
    "python_stdio": """\
Language: Python 3 competitive-programming style (stdin -> stdout).
- "prompt": a full problem statement with "Input" and "Output" sections and constraints. The correct output must be unique (no "print any valid answer").
- "solution": a complete Python 3 program that reads standard input and prints the answer.
- "test": a JSON ARRAY (not a string) of 8-12 objects {{"input": "...", "output": "..."}} with exact expected output, including edge cases and a couple of larger inputs (each input under 5000 characters). Outputs must differ between tests.""",
    "repo_python": """\
Task type: fix a bug in a small multi-file Python package.
- "files": an object mapping 2-4 relative paths (e.g. "billing/__init__.py", "billing/tax.py") to the FULL contents of each file. The package contains exactly one realistic bug (it may involve more than one file).
- "prompt": a GitHub-issue-style bug report describing the wrong behaviour (do NOT reveal the fix), ending with: Fix it using `### FILE: <path>` and `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` blocks.
- "test": Python `assert` statements (6-12) that import from the package (e.g. `from billing.tax import apply_tax`) and FAIL on the buggy code but pass after the fix.
- "solution": the fix as `### FILE: <path>` followed by `<<<<<<< SEARCH`, the exact original lines, `=======`, the replacement lines, `>>>>>>> REPLACE` (one or more blocks; SEARCH text copied exactly from the file).""",
}
LANG_ALIASES = {"py": "python", "js": "javascript", "node": "javascript", "ts": "typescript", "c++": "cpp",
                "sqlite": "sql", "stdio": "python_stdio", "repo": "repo_python"}
FENCE = {"python": "python", "javascript": "javascript", "typescript": "typescript", "cpp": "cpp", "sql": "sql",
         "python_stdio": "python"}
MIN_CHECKS = {"python": 6, "javascript": 6, "typescript": 6, "cpp": 6, "sql": 4, "python_stdio": 6, "repo_python": 4}


def build_prompt(lang: str, n: int, rng: random.Random) -> str:
    topics = [rng.choice(TOPICS) for _ in range(n)] if n > len(TOPICS) else rng.sample(TOPICS, n)
    extra = {"sql": ', "starter_code"', "repo_python": ', "files"'}.get(lang, "")
    return (
        f"Create {n} new, high-quality programming tasks for training and testing a coding model.\n\n"
        + COMMON_RULES.format(n=n, lang=lang, extra_keys=extra, topics="; ".join(topics))
        + "\n" + LANG_RULES[lang] + "\n"
    )


def copy_to_clipboard(text: str) -> bool:
    try:
        import tkinter

        r = tkinter.Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(text)
        r.update()
        r.destroy()
        return True
    except Exception:
        return False


def read_clipboard() -> str:
    import tkinter

    r = tkinter.Tk()
    r.withdraw()
    try:
        return r.clipboard_get()
    finally:
        r.destroy()


def parse_reply(text: str) -> list[dict]:
    """Pull every JSON object with a "prompt" key out of the reply (array, JSON lines, or pretty-printed)."""
    dec, out, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict) and "prompt" in obj:
            out.append(obj)
            i = end
        else:
            i += 1
    return out


def count_checks(lang: str, test: str) -> int:
    if lang == "python_stdio":
        return len(json.loads(test))
    if lang in ("python", "sql", "repo_python"):
        try:
            return sum(isinstance(n, ast.Assert) for n in ast.parse(test).body)
        except SyntaxError:
            return 0
    if lang == "cpp":
        return sum(bool(_CPP_ASSERT_LINE.match(l)) for l in test.splitlines())
    return sum(l.strip().startswith(("assert(", "assert.")) for l in test.splitlines())


def normalize(raw: dict) -> tuple[dict | None, str]:
    lang = str(raw.get("lang", "")).strip().lower()
    lang = LANG_ALIASES.get(lang, lang)
    if lang not in LANG_RULES:
        return None, f"unknown lang {raw.get('lang')!r}"
    prompt, solution, test = raw.get("prompt"), raw.get("solution"), raw.get("test")
    if not (isinstance(prompt, str) and prompt.strip() and isinstance(solution, str) and solution.strip() and test):
        return None, "missing prompt/solution/test"
    if lang == "python_stdio":
        cases = test if isinstance(test, list) else json.loads(test) if isinstance(test, str) else None
        if not isinstance(cases, list) or not all(isinstance(c, dict) and "input" in c and "output" in c for c in cases):
            return None, "stdio test must be a list of {input, output}"
        cases = [{"input": str(c["input"]), "output": str(c["output"])} for c in cases]
        if len({c["output"].strip() for c in cases}) < 2:
            return None, "stdio outputs are all identical (a constant answer would pass)"
        test = json.dumps(cases)
    elif not isinstance(test, str):
        return None, "test must be a string"
    task = {"category": "generated", "lang": "python" if lang == "repo_python" else lang, "prompt": prompt.strip(),
            "test": test, "topic": str(raw.get("topic", ""))}
    if lang == "repo_python":
        files = raw.get("files")
        if not isinstance(files, dict) or not files or not all(isinstance(v, str) for v in files.values()):
            return None, "repo task needs files: {path: content}"
        if "<<<<<<< SEARCH" not in solution:
            return None, "repo solution must be SEARCH/REPLACE blocks"
        task["files"], task["oracle_patch"] = files, solution
    else:
        task["oracle_solution"] = solution if "```" in solution else f"```{FENCE[lang]}\n{solution.strip()}\n```"
    if lang == "sql":
        if not isinstance(raw.get("starter_code"), str) or not raw["starter_code"].strip():
            return None, "sql task needs starter_code (schema + seed rows)"
        task["starter_code"] = raw["starter_code"]
    task["kind"] = lang
    task["id"] = f"gen_{lang}_{hashlib.sha1(task['prompt'].encode('utf-8')).hexdigest()[:12]}"
    return task, "ok"


def check(task: dict) -> tuple[bool, str]:
    kind, lang = task["kind"], task["lang"]
    n = count_checks(kind, task["test"])
    if n < MIN_CHECKS[kind]:
        return False, f"too few checks ({n} < {MIN_CHECKS[kind]})"
    sol = task.get("oracle_patch") or task["oracle_solution"]
    score, info = verify(sol, task["test"], task.get("starter_code", ""), task.get("files"), lang, timeout=8.0, use_cache=False)
    if score != 1.0:
        return False, f"reference solution fails its tests ({score}): {str(info.get('status'))[:150]}"
    if kind == "repo_python":  # the unfixed package must actually fail the tests
        p, src = max(task["files"].items(), key=lambda kv: len(kv[1]))  # a no-op edit on a non-empty file
        noop = f"### FILE: {p}\n<<<<<<< SEARCH\n{src}\n=======\n{src}\n>>>>>>> REPLACE"
        before, _ = verify(noop, task["test"], files=task["files"], lang="python", timeout=8.0, use_cache=False)
        if before >= 1.0:
            return False, "tests already pass on the buggy code (no real bug)"
    if kind == "python":
        from fuzz_tests import audit_test_strength

        rep = audit_test_strength(task)
        if not rep["airtight"]:
            return False, "tests can be passed by a hard-coded sham solution"
    return True, "ok"


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cmd_ingest(args):
    if args.clipboard:
        text = read_clipboard()
    else:
        paths = []
        for p in args.files:
            paths += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith((".json", ".jsonl", ".txt", ".md"))) if os.path.isdir(p) else [p]
        text = "\n".join(open(p, encoding="utf-8").read() for p in paths)
    raws = parse_reply(text)
    if not raws:
        sys.exit("No cases found. Copy the app's whole reply (the ```json block) and try again.")
    existing = load_jsonl(args.out)
    reference = existing + [r for p in args.dedupe_against.split(",") if p for r in load_jsonl(p.strip())]
    ref_grams = [char_ngrams(extract_row_text(r)) for r in reference]
    seen_ids = {r.get("id") for r in existing}

    tasks, rejected = [], []
    for raw in raws:
        task, why = normalize(raw)
        if task is None:
            rejected.append({"why": why, "case": raw})
            print(f"  drop  {str(raw.get('lang', '?')):12s} {str(raw.get('topic', ''))[:28]:28s} {why[:90]}")
        elif task["id"] in seen_ids:
            rejected.append({"why": "duplicate (same prompt already kept)", "case": raw})
        else:
            seen_ids.add(task["id"])
            tasks.append(task)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check, tasks))

    kept = []
    for task, (ok, why) in zip(tasks, results):
        if ok:
            g = char_ngrams(extract_row_text(task))
            if any(jaccard(g, rg) >= 0.6 for rg in ref_grams):
                ok, why = False, "near-duplicate of an existing case"
            else:
                ref_grams.append(g)
        if ok:
            task["source"] = f"generated:{args.source}" if args.source else "generated"
            kept.append(task)
        else:
            rejected.append({"why": why, "case": task})
        print(f"  {'KEEP' if ok else 'drop'}  {task['kind']:12s} {task.get('topic', '')[:28]:28s} {'' if ok else why[:90]}")

    with open(args.out, "a", encoding="utf-8") as f:
        for t in kept:
            f.write(json.dumps(t) + "\n")
    with open(args.rejected, "a", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r) + "\n")
    print(f"\nFound {len(raws)} cases: kept {len(kept)}, dropped {len(raws) - len(kept)}. "
          f"{args.out} now has {len(existing) + len(kept)} cases.")
    if rejected:
        print(f"Why cases were dropped: {args.rejected}")


def cmd_stats(args):
    import collections

    kept = load_jsonl(args.out)
    print(f"{args.out}: {len(kept)} verified cases")
    for kind, n in collections.Counter(t.get("kind", t.get("lang")) for t in kept).most_common():
        print(f"  {kind:14s} {n}")
    rej = load_jsonl(args.rejected)
    if rej:
        print(f"\n{args.rejected}: {len(rej)} dropped. Most common reasons:")
        reasons = collections.Counter(re.sub(r"\(.*", "", r.get("why", "")).strip()[:70] for r in rej)
        for why, n in reasons.most_common(8):
            print(f"  {n:4d}  {why}")


def cmd_recheck(args):
    """Re-verify every kept case from scratch (catches cases added or edited without going through ingest)."""
    kept = load_jsonl(args.out)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check, kept))
    good = [t for t, (ok, _) in zip(kept, results) if ok]
    for t, (ok, why) in zip(kept, results):
        if not ok:
            print(f"  removed {t.get('id')}: {why[:100]}")
    with open(args.out, "w", encoding="utf-8") as f:
        for t in good:
            f.write(json.dumps(t) + "\n")
    print(f"Rechecked {len(kept)} cases: {len(good)} still pass, {len(kept) - len(good)} removed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompt", help="print (and copy) a generation request to paste into the app")
    p.add_argument("--lang", default="python", help="python | javascript | typescript | cpp | sql | python_stdio | repo_python")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-file", default="", help="also write the request to this file")
    for name, helptext in (("stats", "counts per language + most common drop reasons"),
                           ("recheck", "re-verify every kept case; remove any that no longer pass")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--out", default="tasks_generated.jsonl")
        s.add_argument("--rejected", default="cases_rejected.jsonl")
        s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    i = sub.add_parser("ingest", help="verify the app's reply and keep the good cases")
    i.add_argument("files", nargs="*", help="reply files and/or folders of them (or use --clipboard)")
    i.add_argument("--clipboard", action="store_true")
    i.add_argument("--source", default="", help="which model wrote them, e.g. deepseek-v4.1-flash")
    i.add_argument("--out", default="tasks_generated.jsonl")
    i.add_argument("--rejected", default="cases_rejected.jsonl")
    i.add_argument("--dedupe-against", default="train_clean.jsonl,tasks.jsonl")
    i.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    if args.cmd == "prompt":
        lang = LANG_ALIASES.get(args.lang.lower(), args.lang.lower())
        if lang not in LANG_RULES:
            sys.exit(f"--lang must be one of: {', '.join(LANG_RULES)}")
        text = build_prompt(lang, args.n, random.Random(args.seed))
        print(text)
        if args.out_file:
            with open(args.out_file, "w", encoding="utf-8") as f:
                f.write(text)
        elif copy_to_clipboard(text):
            print("(copied to clipboard: paste it into the app)", file=sys.stderr)
    elif args.cmd == "stats":
        cmd_stats(args)
    elif args.cmd == "recheck":
        cmd_recheck(args)
    else:
        if not args.clipboard and not args.files:
            sys.exit("Give reply files / a folder, or --clipboard.")
        cmd_ingest(args)


if __name__ == "__main__":
    main()
