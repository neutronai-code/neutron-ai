"""Import public competitive-programming problems as stdin/stdout tasks (`lang="python_stdio"`).

Source: agentica-org/DeepCoder-Preview-Dataset (MIT) — `primeintellect` (16k) + `taco` (7k) train configs.
The `lcbv5` config is LiveCodeBench problems and is deliberately NOT imported, so LiveCodeBench stays a clean
benchmark for Hadron. Function-call (`fn_name`) problems are skipped; only stdin/stdout problems are kept.

Each parquet file (~100-470 MB, mostly huge test inputs) is downloaded, converted, and deleted, so disk use stays
low. Each problem keeps at most --max-tests tests (no single input/output over --max-test-bytes). With
--verify-oracle (default on) the dataset's own reference solution must pass the kept tests, which drops problems
whose tests are broken before you spend teacher calls on them.

Usage:
  uv run --with pyarrow import_tasks.py                      # everything -> tasks_deepcoder.jsonl
  uv run --with pyarrow import_tasks.py --limit 200          # quick sample
"""
import argparse, concurrent.futures, hashlib, json, os, sys, time, urllib.request

REPO = "agentica-org/DeepCoder-Preview-Dataset"
PROMPT_PREFIX = "Solve the following coding problem using the programming language python:"


def list_files(config: str) -> list[str]:
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main/{config}", timeout=60) as r:
        return sorted(f["path"] for f in json.loads(r.read()) if f["type"] == "file" and f["path"].endswith(".parquet")
                      and f["path"].split("/")[-1].startswith("train-"))


def download(path: str, dest_dir: str) -> str:
    dest = os.path.join(dest_dir, path.replace("/", "__"))
    if os.path.exists(dest):
        return dest
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done, last = 0, 0.0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 2:
                last = time.time()
                print(f"\r  downloading {path}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    os.replace(tmp, dest)
    print(f"\r  downloaded {path} ({done / 1e6:.0f} MB)" + " " * 20, flush=True)
    return dest


def pick(tests: list, k: int) -> list:
    if len(tests) <= k:
        return tests
    head, rest = tests[:4], tests[4:]
    step = len(rest) / (k - 4)
    return head + [rest[int(i * step)] for i in range(k - 4)]


def convert(config: str, row: dict, max_tests: int, max_bytes: int) -> dict | None:
    problem = (row.get("problem") or "").strip()
    raw = row.get("tests")
    if not problem or raw is None:
        return None
    try:
        tests = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    pairs = []
    if isinstance(tests, list):  # primeintellect: [{"type": "stdin_stdout", "input": ..., "output": ...}]
        if not all(isinstance(t, dict) and t.get("type", "stdin_stdout") == "stdin_stdout" for t in tests):
            return None
        pairs = [(t.get("input"), t.get("output")) for t in tests]
    elif isinstance(tests, dict):  # taco: {"inputs": [...], "outputs": [...], "fn_name"?: ...}
        if tests.get("fn_name"):
            return None
        pairs = list(zip(tests.get("inputs") or [], tests.get("outputs") or []))
    pairs = [(i, o) for i, o in pairs if isinstance(i, str) and isinstance(o, str) and len(i) <= max_bytes and len(o) <= max_bytes]
    if len(pairs) < 2:
        return None
    sols = row.get("solutions")
    try:
        sols = json.loads(sols) if isinstance(sols, str) else (list(sols) if sols is not None else [])
    except ValueError:
        sols = []
    oracle = next((s for s in sols if isinstance(s, str) and s.strip()), "")
    if oracle and "```" not in oracle:
        oracle = f"```python\n{oracle.strip()}\n```"
    if problem.startswith(PROMPT_PREFIX):
        problem = problem[len(PROMPT_PREFIX):].strip()
    return {
        "id": f"dc_{config}_{hashlib.sha1(problem.encode('utf-8')).hexdigest()[:12]}",
        "category": "competitive",
        "lang": "python_stdio",
        "source": f"{REPO}/{config}",
        "prompt": problem,
        "test": json.dumps([{"input": i, "output": o} for i, o in pick(pairs, max_tests)]),
        "oracle_solution": oracle,
    }


def oracle_ok(task: dict) -> bool:
    from sandbox import verify

    if not task["oracle_solution"]:
        return False
    score, _ = verify(task["oracle_solution"], task["test"], lang="python_stdio", timeout=6.0, use_cache=False)
    return score == 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tasks_deepcoder.jsonl")
    ap.add_argument("--configs", default="primeintellect,taco")
    ap.add_argument("--max-tests", type=int, default=12)
    ap.add_argument("--max-test-bytes", type=int, default=100_000)
    ap.add_argument("--limit", type=int, default=0, help="stop after N kept tasks (0 = all)")
    ap.add_argument("--no-verify-oracle", action="store_true", help="skip checking tests against the reference solution")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--cache-dir", default=".cache_import")
    ap.add_argument("--keep-downloads", action="store_true")
    args = ap.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is needed:  uv run --with pyarrow import_tasks.py")

    os.makedirs(args.cache_dir, exist_ok=True)
    seen, kept, dropped_tests, dropped_format = set(), 0, 0, 0
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            seen = {json.loads(l)["id"] for l in f if l.strip()}
        print(f"Resuming: {len(seen)} tasks already in {args.out}")
    existing = len(seen)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    with open(args.out, "a", encoding="utf-8") as out:
        for config in [c.strip() for c in args.configs.split(",") if c.strip()]:
            if config.startswith("lcb"):
                print(f"Skipping {config}: LiveCodeBench problems would contaminate your LiveCodeBench evals.")
                continue
            for path in list_files(config):
                local = download(path, args.cache_dir)
                pf = pq.ParquetFile(local)
                cols = [c for c in ("problem", "tests", "solutions") if c in pf.schema_arrow.names]
                for batch in pf.iter_batches(batch_size=128, columns=cols):
                    tasks = []
                    for row in batch.to_pylist():
                        t = convert(config, row, args.max_tests, args.max_test_bytes)
                        if t is None:
                            dropped_format += 1
                        elif t["id"] not in seen:
                            seen.add(t["id"])
                            tasks.append(t)
                    ok = list(pool.map(oracle_ok, tasks)) if not args.no_verify_oracle else [True] * len(tasks)
                    for t, good in zip(tasks, ok):
                        if not good:
                            dropped_tests += 1
                            continue
                        out.write(json.dumps(t) + "\n")
                        kept += 1
                    out.flush()
                    print(f"\r  {config}: kept {kept} | skipped {dropped_format} (format) + {dropped_tests} (reference solution failed tests)", end="", flush=True)
                    if args.limit and kept >= args.limit:
                        break
                print()
                if not args.keep_downloads:
                    os.remove(local)
                if args.limit and kept >= args.limit:
                    break
            if args.limit and kept >= args.limit:
                break
    pool.shutdown()
    print(f"\nWrote {kept} new tasks -> {args.out}  (total now {existing + kept})")


if __name__ == "__main__":
    main()
