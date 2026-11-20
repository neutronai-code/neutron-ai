"""Quark v0.3 data: fill-in-the-middle holes cut from real open-source repos, checked by each repo's OWN test suite.

The repos come from quark_repos.jsonl (permissive licenses, created after Qwen2.5-Coder's pretraining cutoff, tests
verified by the repo-picking agent). For each repo, at its pinned commit, on Linux:

  1. install it and run its tests once (they must pass: that is the "original code passes" check),
  2. cut candidate holes in its source files (same span rules as fim.py: short 1-6 line statements and long 3-15 line
     blocks / function bodies),
  3. replace the hole with a stub and re-run the tests: keep the hole only if they now FAIL (the code really matters),
  4. restore the file, and finally re-run the tests to make sure the checkout is clean.

A file whose first few holes all still pass gets skipped (its code isn't tested). Each row stores a window of the file
around the hole (the prompt) plus the line range, so a fill can later be spliced into the real file and re-tested.

Runs on Modal (Linux, one container per repo): modal run modal_repo_fim.py
"""
import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path

from fim import brace_spans, format_qwen_fim_prompt, python_spans

PREFIX_CHARS, SUFFIX_CHARS = 2400, 900  # the prompt window; train.py keeps prompts up to ~3,584 chars
MISS_LIMIT = 3  # skip a file after this many holes in a row that the tests don't notice

PY_SETUP = "uv sync -q --frozen --all-extras --all-groups"
PY_TEST = ".venv/bin/python -m pytest -x -q -p no:cacheprovider"
PNPM = "corepack enable && pnpm install --frozen-lockfile --ignore-scripts"
CMAKE = "cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug"
CTEST = "cmake --build build && ctest --test-dir build --stop-on-failure --output-on-failure"

# Linux commands for each repo. `src` is a list of globs for the hand-written source files holes may be cut from.
REPOS = {
    "TanStack/pacer": dict(setup=PNPM, test="cd packages/pacer && npx vitest run --bail=1",
                           src=["packages/pacer/src/**/*.ts"]),
    "serkodev/markdown-exit": dict(setup=PNPM, test="npx vitest run --bail=1",
                                   src=["packages/markdown-exit/src/**/*.ts"]),
    "ddkbs/sketchy": dict(setup="npm ci --ignore-scripts && npm run build", test="npx vitest run --bail=1",
                          src=["src/**/*.ts"]),
    "meodai/heerich": dict(setup="npm ci --ignore-scripts",
                           test=" && ".join(f"node tests/{t}.test.js" for t in
                                            ("faces", "update", "fit", "slopes", "clip", "site-draw", "site-orbit")),
                           src=["src/*.js"]),
    # The H.264 proxy test skips itself without ffmpeg (as when the holes were mined); with ffmpeg in the image it
    # scores 33.5 dB against its 35 dB bar on Modal, so the baseline failed and only exact fills could pass.
    "ronak-create/FableCut": dict(setup="true",
                                  test="node --test --test-skip-pattern='frame-accurate H.264 proxy'", src=["*.js"]),
    "raulcd/datanomy": dict(setup=PY_SETUP, test=PY_TEST, src=["src/datanomy/**/*.py"]),
    "EbodShojaei/bake": dict(setup="uv venv -q && uv pip install -q -e . pytest", test=PY_TEST,
                             src=["mbake/**/*.py"]),
    "MichielMe/fastscheduler": dict(  # this timing test is flaky on a busy 2-CPU container: it fails ~1 run in 20
        setup=PY_SETUP, src=["src/fastscheduler/**/*.py"],
        test=PY_TEST + " --deselect tests/test_fastapi_integration.py::TestIntegrationScenarios::"
                       "test_async_jobs_with_fastapi"),
    "mosaico-labs/refx": dict(setup=f"{CMAKE} -DBUILD_TESTING=ON && cmake --build build", test=CTEST,
                              src=["src/**/*.h", "src/**/*.hpp", "src/**/*.cpp"]),
    "EmberEmu/Hexi": dict(setup=f"{CMAKE} -DENABLE_TESTING=ON && cmake --build build", test=CTEST,
                          src=["include/hexi/**/*.h"]),

    # --- second batch: the other 30 verified repos ---
    # TypeScript libraries
    "rhinobase/hono-openapi": dict(setup=PNPM, test="npx vitest run --bail=1", src=["src/*.ts"]),
    "cloudflare/workers-oauth-provider": dict(setup="npm ci --ignore-scripts", test="npx vitest run --bail=1",
                                              src=["src/**/*.ts"]),
    "TanStack/hotkeys": dict(setup=PNPM, test="cd packages/hotkeys && npx vitest run --bail=1",
                             src=["packages/hotkeys/src/**/*.ts"]),
    "spaansba/ForesightJS": dict(setup=PNPM, test="cd packages/js.foresight && npx vitest run --bail=1",
                                 src=["packages/js.foresight/src/**/*.ts"]),
    "affaan-m/agentshield": dict(setup="npm ci --ignore-scripts", test="npm test", src=["src/**/*.ts"]),
    "yamlresume/yamlresume": dict(setup=f"{PNPM} && pnpm -r build",
                                  test="cd packages/core && npx vitest run --bail=1",
                                  src=["packages/core/src/**/*.ts"]),
    "googleapis/js-genai": dict(setup="npm ci --ignore-scripts", test="npm run unit-test", src=["src/**/*.ts"]),
    # React / Next.js apps: no unit tests, so the check is the type checker (good for SFT data, weak for RL)
    "ibelick/zola": dict(setup="npm ci", test="npx tsc --noEmit",
                         src=[f"{d}/**/*.{e}" for d in ("app", "components", "lib", "utils") for e in ("ts", "tsx")]),
    "hristo2612/SQLNoir": dict(setup="npm ci", test="npx tsc --noEmit", src=["src/**/*.ts", "src/**/*.tsx"]),
    "jhlee0409/claude-code-history-viewer": dict(setup=PNPM, test="npx tsc --noEmit -p tsconfig.app.json",
                                                 src=["src/**/*.ts", "src/**/*.tsx"]),
    "DayuanJiang/next-ai-draw-io": dict(
        setup="npm ci --ignore-scripts", test="npx tsc --noEmit",
        src=[f"{d}/**/*.{e}" for d in ("app", "components", "contexts", "hooks", "lib", "stores")
             for e in ("ts", "tsx")]),
    # JavaScript libraries
    "law-chain-hot/websocket-devtools": dict(setup=PNPM, test="node --test", src=["src/**/*.js"]),
    "humanwhocodes/crosspost": dict(setup="npm ci --ignore-scripts && npm run build",
                                    test="npm run build && npx mocha 'tests/**/*.{js,mjs}' --exit --forbid-only --bail",
                                    src=["src/**/*.js", "src/**/*.ts"]),
    "pompelmi/pompelmi": dict(setup="npm ci --ignore-scripts", test="npm test", src=["src/*.js"]),
    "petergyang/human-review": dict(setup="npm ci --ignore-scripts",
                                    test="node --test --test-force-exit --test-timeout=60000", src=["src/*.js"]),
    # Python libraries
    "knowsuchagency/mcp2cli": dict(setup="uv venv -q && uv pip install -q -e '.[test]'",  # TOON needs a CLI
                                   test=PY_TEST + " -k 'not TestToonEncode'",
                                   src=["src/mcp2cli/**/*.py"]),
    "gradio-app/trackio": dict(setup="uv venv -q && uv pip install -q -e '.[dev]' pytest pyarrow",
                               test=PY_TEST + " tests/unit --ignore=tests/unit/test_gpu_hardware.py",
                               src=["trackio/**/*.py"]),
    "microsoft/RAMPART": dict(setup="uv venv -q && uv pip install -q -e . --group dev aiohttp",
                              test=PY_TEST + " tests/unit", src=["rampart/**/*.py"]),
    "ozeranskii/httptap": dict(setup="uv venv -q && uv pip install -q -e . --group test", test=PY_TEST,
                               src=["httptap/**/*.py"]),
    "saidsurucu/borsapy": dict(setup="uv venv -q && uv pip install -q -e '.[dev]'",  # backtests fetch live rates
                               test=PY_TEST + " --ignore=tests/test_backtest.py",
                               src=["borsapy/**/*.py"]),
    "cayleypy/cayleypy": dict(setup="uv venv -q && uv pip install -q -e '.[test]'", test=PY_TEST,
                              src=["cayleypy/**/*.py"]),
    "offerrall/FuncToWeb": dict(setup="uv venv -q && uv pip install -q -e '.[test,test-outputs]'",
                                test=PY_TEST + " tests/unit tests/integration", src=["src/func_to_web/**/*.py"]),
    "docling-project/docling-mcp": dict(setup="uv venv -q && uv pip install -q -e . --group dev beautifulsoup4",
                                        test=PY_TEST, src=["docling_mcp/**/*.py"]),
    "oracle/agent-spec": dict(
        setup="uv venv -q && uv pip install -q -e ./pyagentspec pytest pytest-asyncio cryptography httpx requests "
              "fastapi anyio",
        test="SKIP_LLM_TESTS=1 " + PY_TEST + " -o asyncio_default_fixture_loop_scope=function "
             "pyagentspec/tests --ignore=pyagentspec/tests/adapters "
             "--ignore=pyagentspec/tests/evaluation",
        src=["pyagentspec/src/**/*.py"]),
    # C++ libraries
    "ashvardanian/ForkUnion": dict(
        setup=f"{CMAKE} -DFORKUNION_BUILD_TEST=ON -DFORKUNION_TEST_SKIP_STRESS=ON && cmake --build build",
        test=CTEST + " -E forkunion_test_c11",  # GCC nested-functions test segfaults on Linux
        src=["include/**/*.hpp", "include/**/*.h"]),
    "0xeb/fastmcpp": dict(setup=f"{CMAKE} -DFASTMCPP_BUILD_EXAMPLES=OFF && cmake --build build", test=CTEST + " -j 2",
                          src=["include/**/*.hpp", "src/**/*.cpp"]),
    "nvpro-samples/nv_cluster_lod_builder": dict(
        setup=f"git submodule update --init --recursive --depth 1 && {CMAKE} -DBUILD_TESTING=ON "
              "-DCMAKE_CXX_STANDARD=20 -DBUILD_NV_CLUSTER_BUILDER_TESTING=OFF && cmake --build build",
        test=CTEST, src=["include/**/*.h", "include/**/*.hpp", "src/**/*.cpp", "src/**/*.hpp"]),
    # FinalScave/SweetEditor: left out, one of its tests segfaults on Linux (g++ 14) at the pinned commit
    "sentomk/patternia": dict(  # codegen.* tests count instructions, which depends on the exact compiler
        setup=f"{CMAKE} -DPTN_BUILD_TESTS=ON -DPTN_BUILD_EXAMPLES=OFF && cmake --build build",
        test=CTEST + " -E '^codegen'", src=["include/**/*.hpp", "include/**/*.h"]),
    "librats/librats": dict(
        setup=f"{CMAKE} -DRATS_BUILD_TESTS=ON -DRATS_BUILD_CLIENT=OFF -DRATS_BUILD_EXAMPLES=OFF && cmake --build build",
        test=CTEST + " -j 2 -E IOPollerTest.DetectsReadableMultipleSockets",  # fails in Modal's sandbox
        src=["src/**/*.cpp", "src/**/*.h"]),
}
EXT_LANG = {".py": "python", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript", ".mjs": "javascript",
            ".h": "cpp", ".hpp": "cpp", ".cpp": "cpp", ".cc": "cpp"}


def load_repo_info(path: str = "quark_repos.jsonl") -> dict[str, dict]:
    return {r["repo"]: r for r in (json.loads(line) for line in open(path, encoding="utf-8") if line.strip())}


def run(cmd: str, cwd: str, timeout: float) -> tuple[bool, float, str]:
    """Run a shell command; (passed, seconds, last lines of output). Kills the whole process group on timeout."""
    t0 = time.time()
    proc = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True, text=True, errors="replace")
    try:
        out, _ = proc.communicate(timeout=timeout)
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        ok, out = False, (out or "") + f"\n[timeout after {timeout:.0f}s]"
    try:  # anything the command left running in the background (a test server, a worker): on a long-lived machine
        os.killpg(proc.pid, signal.SIGKILL)  # these would pile up, unlike in a throwaway container
    except (ProcessLookupError, PermissionError):
        pass
    return ok, time.time() - t0, "\n".join((out or "").splitlines()[-15:])


def splice(lines: list[str], lo: int, hi: int, middle: str) -> str:
    """The file with lines [lo, hi) replaced by `middle` (same convention as fim.py: suffix starts with a newline)."""
    return "".join(lines[:lo]) + middle + "\n" + "".join(lines[hi:])


def stub_for(lang: str, lines: list[str], lo: int) -> str:
    if lang == "python":  # an empty Python block is a syntax error; `pass` at the hole's indent keeps it valid
        first = lines[lo]
        return first[:len(first) - len(first.lstrip())] + "pass"
    return ""  # brace languages: just remove the lines (the spans are brace-balanced)


def window(lines: list[str], lo: int, hi: int) -> tuple[str, str]:
    """Prompt prefix/suffix: up to PREFIX_CHARS before the hole and SUFFIX_CHARS after it, cut at line boundaries."""
    prefix = "".join(lines[:lo])
    if len(prefix) > PREFIX_CHARS:
        prefix = prefix[-PREFIX_CHARS:]
        prefix = prefix[prefix.find("\n") + 1:]
    suffix = "\n" + "".join(lines[hi:])
    if len(suffix) > SUFFIX_CHARS:
        suffix = suffix[:suffix.rfind("\n", 0, SUFFIX_CHARS) + 1]
    return prefix, suffix


def source_files(root: Path, globs: list[str]) -> list[Path]:
    files = set()
    for g in globs:
        for p in root.glob(g):
            rel = p.relative_to(root).as_posix()
            if (p.is_file() and p.suffix in EXT_LANG and not p.name.endswith(".d.ts")
                    and not any(part in ("test", "tests", "__tests__", "node_modules", "dist", "build")
                                for part in Path(rel).parts[:-1])
                    and ".test." not in p.name and ".spec." not in p.name and not p.name.startswith("test_")):
                files.add(p)
    return sorted(files)


def candidates(text: str, lang: str, seed: str) -> list[tuple[int, int, str]]:
    """Non-overlapping hole spans for one file, short and long ones interleaved, in a seeded random order."""
    lines = text.splitlines(keepends=True)
    if not lines or max(len(ln) for ln in lines) > 400:  # minified / generated
        return []
    try:
        short = python_spans(text) if lang == "python" else brace_spans(lines)
        long = python_spans(text, hard=True) if lang == "python" else brace_spans(lines, hard=True)
    except SyntaxError:
        return []
    rng = random.Random(seed)
    rng.shuffle(short)
    rng.shuffle(long)
    picked, used = [], []
    kinds = [(long, "long"), (short, "short")]
    rng.shuffle(kinds)  # half the files lead with a short hole, so the first pass over a repo isn't all long ones
    for i in range(max(len(short), len(long))):
        for spans, size in kinds:
            if i < len(spans):
                lo, hi = spans[i]
                if "".join(lines[lo:hi]).strip() and not any(lo < h and l < hi for l, h in used):
                    used.append((lo, hi))
                    picked.append((lo, hi, size))
    return picked


def mine_repo(repo: str, root: str, setup: str, test: str, src: list[str], info: dict, max_holes: int = 40,
              minutes: float = 45, log=print) -> dict:
    """Install + baseline-test the checkout at `root`, then cut and verify holes. Returns {"rows", "stats"}."""
    rootp = Path(root)
    stats = {"repo": repo, "status": "", "setup_s": 0.0, "test_s": 0.0, "checked": 0, "kept": 0, "timeouts": 0,
             "files": 0, "files_skipped": 0}
    ok, secs, tail = run(setup, root, 30 * 60)
    stats["setup_s"] = round(secs)
    if not ok:
        stats["status"] = "setup failed"
        log(f"[{repo}] SETUP FAILED ({secs:.0f}s)\n{tail}")
        return {"rows": [], "stats": stats, "tail": tail}
    ok, secs, tail = run(test, root, 15 * 60)
    if not ok:  # one retry in case of a flaky test
        ok, secs, tail = run(test, root, 15 * 60)
    stats["test_s"] = round(secs, 1)
    if not ok:
        stats["status"] = "tests fail on Linux"
        log(f"[{repo}] BASELINE TESTS FAIL ({secs:.0f}s)\n{tail}")
        return {"rows": [], "stats": stats, "tail": tail}
    log(f"[{repo}] setup {stats['setup_s']}s, tests pass in {secs:.1f}s")
    check_timeout = max(60.0, 3 * secs + 30)

    queues = {}
    for p in source_files(rootp, src):
        text = p.read_text(encoding="utf-8", errors="replace")
        lang = EXT_LANG[p.suffix]
        rel = p.relative_to(rootp).as_posix()
        if spans := candidates(text, lang, f"{repo}:{rel}"):
            queues[rel] = {"text": text, "lang": lang, "spans": spans, "misses": 0}
    stats["files"] = len(queues)
    order = list(queues)
    random.Random(repo).shuffle(order)
    log(f"[{repo}] {len(order)} source files with candidate holes")

    rows, deadline = [], time.time() + minutes * 60
    while order and len(rows) < max_holes and time.time() < deadline:
        for rel in list(order):  # round-robin: one hole per file per pass, so holes spread over the repo
            if len(rows) >= max_holes or time.time() >= deadline:
                break
            q = queues[rel]
            if not q["spans"] or q["misses"] >= MISS_LIMIT:
                stats["files_skipped"] += q["misses"] >= MISS_LIMIT
                order.remove(rel)
                continue
            lo, hi, size = q["spans"].pop(0)
            lines = q["text"].splitlines(keepends=True)
            path = rootp / rel
            try:
                path.write_text(splice(lines, lo, hi, stub_for(q["lang"], lines, lo)), encoding="utf-8")
                passed, secs, tail = run(test, root, check_timeout)
            finally:
                path.write_text(q["text"], encoding="utf-8")
            stats["checked"] += 1
            if passed:  # the tests don't notice this hole: useless as training data
                q["misses"] += 1
                continue
            q["misses"] = 0
            timed_out = "[timeout after" in tail
            stats["timeouts"] += timed_out
            prefix, suffix = window(lines, lo, hi)
            oracle = "".join(lines[lo:hi]).rstrip("\r\n")
            rows.append({
                "id": f"repo_{repo.replace('/', '__')}_{rel}_L{lo + 1}{'_H' if size == 'long' else ''}",
                "category": "repo_fim", "repo": repo, "kind": info.get("kind", "library"),
                "commit": info.get("commit", ""),
                "license": info.get("license", ""), "path": rel, "lang": q["lang"], "size": size,
                "lo": lo, "hi": hi, "prefix": prefix, "suffix": suffix, "oracle_middle": oracle,
                "fim_prompt": format_qwen_fim_prompt(prefix, suffix), "stub_seconds": round(secs, 1),
                "stub_timeout": timed_out,
            })
            log(f"[{repo}] kept {len(rows)}: {rel}:{lo + 1}-{hi} ({size}, {hi - lo} lines, tests failed in {secs:.0f}s)")
    stats["kept"] = len(rows)
    ok, secs, tail = run(test, root, 15 * 60)  # every file was restored, so the suite must pass again
    stats["status"] = "ok" if ok else "tests fail after mining (flaky?)"
    log(f"[{repo}] done: {stats}")
    return {"rows": rows if ok else [], "stats": stats, "tail": "" if ok else tail}


def check_fill(root: str, path: str, lo: int, hi: int, middle: str, test: str, timeout: float) -> tuple[bool, float]:
    """Splice a model's fill into the real file, run the repo's tests, restore the file. For eval / RL rewards."""
    p = Path(root) / path
    original = p.read_text(encoding="utf-8")
    try:
        p.write_text(splice(original.splitlines(keepends=True), lo, hi, middle), encoding="utf-8")
        ok, secs, _ = run(test, root, timeout)
    finally:
        p.write_text(original, encoding="utf-8")
    return ok, secs


def same_code(a: str, b: str) -> bool:
    """Two fills are the same code (ignoring trailing spaces and blank edges): the original is known to pass."""
    norm = lambda s: [ln.rstrip() for ln in s.strip("\r\n").splitlines()]
    return norm(a) == norm(b)


def merge_report(path: str, records: list[dict]) -> list[dict]:
    """Write `records` into the per-hole report at `path`, keeping other models' results and replacing only the
    (model, hole) pairs in `records`. Returns everything now in the report."""
    p = Path(path)
    if p.exists():
        redone = {(x["model"], x["id"]) for x in records}
        old = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
        records = [x for x in old if (x["model"], x["id"]) not in redone] + records
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in records)
    return records


def print_tables(records: list[dict], models: list[str], max_others: int = 0) -> None:
    """Pass rates by language / hole size / repo for `models`, next to every earlier model in `records` that was
    scored on the same holes (only the best `max_others` of them, if set), then each new model's wins and losses
    against those models, hole by hole."""
    import collections

    res = collections.defaultdict(dict)  # model -> hole id -> record
    for x in records:
        res[x["model"]][x["id"]] = x
    ids = sorted(set.intersection(*(set(res[m]) for m in models)))
    others = [m for m in res if m not in models and set(ids) <= set(res[m])]
    if max_others:
        best = sorted(others, key=lambda o: -sum(res[o][i]["pass"] for i in ids))[:max_others]
        others = [o for o in others if o in best]
    shown = others + models

    def table(title, field):
        groups = sorted({res[models[0]][i][field] for i in ids})
        tags = [m.split("/")[-1] for m in shown]
        print(f"\n{title:28}" + "".join(f"{t[-22:]:>24}" for t in tags))  # the end: version / @commit
        for g in groups + ["ALL"]:
            sel = [i for i in ids if g == "ALL" or res[models[0]][i][field] == g]
            cells = "".join(f"{sum(res[m][i]['pass'] for i in sel) / len(sel):>17.1%} ({len(sel):>3})" for m in shown)
            print(f"{str(g)[:28]:28}{cells}")

    table("language", "lang")
    table("hole size", "size")
    table("repo", "repo")
    for m in models:
        if others:
            print(f"\n{m.split('/')[-1]} hole by hole (wins / losses):")
        for o in sorted(others, key=lambda o: -sum(res[o][i]["pass"] for i in ids)):
            wins = sum(res[m][i]["pass"] and not res[o][i]["pass"] for i in ids)
            losses = sum(res[o][i]["pass"] and not res[m][i]["pass"] for i in ids)
            print(f"  vs {o.split('/')[-1]:34} {wins:>4} / {losses:<4}")


# ---------------------------------------------------------------------------------------------------------------------
# SFT warm-up data (Quark v0.3). The answer to an SFT hole is just the original code, so these holes need no test runs:
# we can cut thousands of them. The first 10 repos stay out of training: their 600 test-checked holes are the benchmark.
HELDOUT = ["TanStack/pacer", "serkodev/markdown-exit", "ddkbs/sketchy", "meodai/heerich", "ronak-create/FableCut",
           "raulcd/datanomy", "EbodShojaei/bake", "MichielMe/fastscheduler", "mosaico-labs/refx", "EmberEmu/Hexi"]
SFT_ONLY = {"FinalScave/SweetEditor": ["src/**/*.cpp", "src/**/*.h", "include/**/*.h"]}  # no Linux tests, fine for SFT
MAX_MIDDLE_CHARS = 1200  # ~400 tokens: longer fills don't fit the training sequence


def build_sft(clones: str, out: str, per_repo: int = 300, verified: list[str] = (), generated: str = "",
              n_generated: int = 1000, seed: int = 0) -> int:
    """Write FIM-SFT rows ({id, lang, fim_prompt, oracle_middle, ...}) from the local clones of every training repo,
    plus the test-verified holes in `verified` files and `n_generated` holes from `generated` (never-solved first)."""
    import subprocess as sp

    info = load_repo_info()
    rng = random.Random(seed)
    rows, seen = [], set()

    def add(r):
        key = (r.get("repo", ""), r.get("path", ""), r.get("lo"), r.get("hi")) if r.get("repo") else r["id"]
        if key not in seen and r["oracle_middle"].strip() and len(r["oracle_middle"]) <= MAX_MIDDLE_CHARS:
            seen.add(key)
            rows.append(r)

    for f in verified:  # test-checked holes from the training repos (their stub really breaks the tests)
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["repo"] not in HELDOUT:
                add(r)
    n_verified = len(rows)

    globs = {**{r: c["src"] for r, c in REPOS.items()}, **SFT_ONLY}
    per_repo_counts = {}
    for repo, src in globs.items():
        if repo in HELDOUT or repo not in info:
            continue
        root = Path(clones) / repo.replace("/", "__")
        head = sp.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != info[repo]["commit"]:
            print(f"!! {repo}: local clone is at {head[:10] or 'nothing'}, not the pinned {info[repo]['commit'][:10]}; skipped")
            continue
        queues = []
        for p in source_files(root, src):
            text = p.read_text(encoding="utf-8", errors="replace")
            rel = p.relative_to(root).as_posix()
            spans = candidates(text, EXT_LANG[p.suffix], f"sft:{repo}:{rel}")
            if spans:
                queues.append((rel, text.splitlines(keepends=True), EXT_LANG[p.suffix], spans))
        rng.shuffle(queues)
        before = len(rows)
        while queues and len(rows) - before < per_repo:  # round-robin over files, as in mining
            for q in list(queues):
                rel, lines, lang, spans = q
                if not spans:
                    queues.remove(q)
                    continue
                lo, hi, size = spans.pop(0)
                prefix, suffix = window(lines, lo, hi)
                add({"id": f"sft_{repo.replace('/', '__')}_{rel}_L{lo + 1}{'_H' if size == 'long' else ''}",
                     "category": "repo_fim_sft", "repo": repo, "kind": info[repo].get("kind", "library"),
                     "path": rel, "lang": lang, "size": size, "lo": lo, "hi": hi,
                     "fim_prompt": format_qwen_fim_prompt(prefix, suffix),
                     "oracle_middle": "".join(lines[lo:hi]).rstrip("\r\n")})
                if len(rows) - before >= per_repo:
                    break
        per_repo_counts[repo] = len(rows) - before
    n_repo = len(rows) - n_verified

    n_gen = 0
    if generated and n_generated:  # our generated cases, holes v0.2's predecessor never solved first
        gen = [json.loads(line) for line in open(generated, encoding="utf-8") if line.strip()]
        rng.shuffle(gen)
        gen.sort(key=lambda r: r.get("baseline_pass_rate", 0.5))
        for r in gen[:n_generated]:
            before = len(rows)
            add({k: r[k] for k in ("id", "category", "lang", "fim_prompt", "oracle_middle")})
            n_gen += len(rows) > before

    rng.shuffle(rows)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{len(rows)} SFT rows -> {out}: {n_verified} test-verified repo holes, {n_repo} more repo holes, "
          f"{n_gen} generated holes")
    for repo, n in sorted(per_repo_counts.items()):
        print(f"  {repo:40} {n}")
    return len(rows)


# ---------------------------------------------------------------------------------------------------------------------
# GRPO with test rewards (Quark v0.3). A fill's reward is whether the repo's own tests pass with it spliced in. The tests
# run in modal_repo_fim.RepoChecker: warm CPU containers per repo, each installed once, scoring one fill at a time.
# Holes come in blocks of one repo so its containers stay warm, and the next block's repo is warmed up in advance.
CHECKER_APP, CHECKER_CLS = "neutron-repo-fim", "RepoChecker"  # `modal deploy modal_repo_fim.py` creates it
MAX_TEST_SECONDS = 35  # repos whose test suite takes longer would make every training step wait on them
WARM_CONTAINERS = 12  # at most RepoChecker's max_containers (16)


def build_grpo(out: str, verified: list[str], stats: list[str], block: int = 16,
               max_test_s: float = MAX_TEST_SECONDS, seed: int = 0) -> int:
    """Write the test-checked holes of the training repos in repo blocks: block 1 of every repo, then block 2, ..."""
    test_s = {}
    for f in stats:  # later files win (batch 3 re-mined some batch-2 repos)
        for s in json.load(open(f, encoding="utf-8")):
            if s.get("status") == "ok":
                test_s[s["repo"]] = s.get("test_s", 0)
    by_repo = {}
    for f in verified:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["repo"] in HELDOUT or len(r["oracle_middle"]) > MAX_MIDDLE_CHARS:
                continue
            if r["repo"] in test_s and test_s[r["repo"]] <= max_test_s:
                by_repo.setdefault(r["repo"], {})[(r["path"], r["lo"], r["hi"])] = r
    rng = random.Random(seed)
    queues = []
    for repo in sorted(by_repo):
        holes = list(by_repo[repo].values())
        rng.shuffle(holes)
        queues.append([holes[k:k + block] for k in range(0, len(holes), block)])
    rng.shuffle(queues)
    blocks = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]
    rows = []
    for i, b in enumerate(blocks):
        nxt = blocks[i + 1][0]["repo"] if i + 1 < len(blocks) else ""
        rows += [{k: r[k] for k in ("id", "repo", "path", "lang", "size", "lo", "hi", "fim_prompt", "oracle_middle")}
                 | {"next_repo": nxt} for r in b]
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    skipped = sorted(r for r, t in test_s.items() if t > max_test_s and r not in HELDOUT)
    print(f"{len(rows)} GRPO holes from {len(by_repo)} repos in {len(blocks)} blocks -> {out}"
          f" (skipped for slow tests: {', '.join(skipped) or 'none'})")
    return len(rows)


_warmed: set[str] = set()


def _checker(repo: str):
    import modal

    return modal.Cls.from_name(CHECKER_APP, CHECKER_CLS)(repo=repo)


def repo_grpo_reward_func(completions, repo, path, lo, hi, oracle_middle, next_repo=None, **kwargs) -> list[float]:
    """TRL reward: 1.0 if the repo's tests pass with the fill spliced in, else 0.0. A hole whose test container broke
    gets 0.0 for every fill, so its group teaches nothing instead of teaching wrong answers."""
    from fim import extract_fim_middle

    n = len(completions)
    fills = [extract_fim_middle(c[0]["content"] if isinstance(c, list) else c) for c in completions]
    holes = [(repo[i], path[i], int(lo[i]), int(hi[i])) for i in range(n)]
    scores: list[float | None] = [None] * n
    calls: dict[tuple, list[int]] = {}  # identical fills of the same hole are tested once
    for i in range(n):
        if not fills[i].strip():
            scores[i] = 0.0
        elif same_code(fills[i], oracle_middle[i]):
            scores[i] = 1.0
        else:
            calls.setdefault((*holes[i], fills[i]), []).append(i)

    for nr in set(next_repo or []):  # start the next block's containers now, so they are installed when it begins
        if nr and nr not in _warmed:
            _warmed.add(nr)
            try:
                for _ in range(WARM_CONTAINERS):
                    _checker(nr).warm.spawn()
            except Exception as e:
                print(f"[repo-grpo] could not warm {nr}: {e}", flush=True)

    futures = {}
    for key in calls:
        try:
            futures[key] = _checker(key[0]).score.spawn(*key[1:])
        except Exception as e:
            futures[key] = e
    broken = set()
    for key, fut in futures.items():
        ok = None
        if not isinstance(fut, Exception):
            try:
                ok = fut.get(timeout=30 * 60)
            except Exception as e:
                fut = e
        if ok is None:
            broken.add(key[:4])
            why = fut if isinstance(fut, Exception) else "its container could not install the repo or pass its tests"
            print(f"[repo-grpo] {key[0]} {key[1]}:{key[2] + 1} could not be tested: {why}", flush=True)
            continue
        for i in calls[key]:
            scores[i] = 1.0 if ok else 0.0
    final = [0.0 if holes[i] in broken else s for i, s in enumerate(scores)]
    for h in dict.fromkeys(holes):
        idx = [i for i in range(n) if holes[i] == h]
        exact = sum(1 for i in idx if fills[i].strip() and same_code(fills[i], oracle_middle[i]))
        print(f"[repo-grpo] {h[0]} {h[1]}:{h[2] + 1}  pass {sum(final[i] for i in idx):.0f}/{len(idx)} "
              f"({exact} exact){'  BROKEN' if h in broken else ''}", flush=True)
    return final


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build Quark's FIM-SFT warm-up data from the local repo clones, or the "
                                             "GRPO hole list (`grpo`).")
    ap.add_argument("action", choices=["sft", "grpo"])
    ap.add_argument("--clones", default=str(Path.home() / "quark_repos"))
    ap.add_argument("--out", default="", help="default: fim_sft_v03.jsonl (sft) / repo_grpo.jsonl (grpo)")
    ap.add_argument("--per-repo", type=int, default=300)
    ap.add_argument("--verified", default="repo_holes_b2.jsonl,repo_holes_b3.jsonl")
    ap.add_argument("--stats", default="repo_holes_b2_stats.json,repo_holes_b3_stats.json")
    ap.add_argument("--generated", default="fim_train_v2.jsonl")
    ap.add_argument("--n-generated", type=int, default=1000)
    a = ap.parse_args()
    verified = [f for f in a.verified.split(",") if f and Path(f).exists()]
    if a.action == "grpo":
        build_grpo(a.out or "repo_grpo.jsonl", verified, [f for f in a.stats.split(",") if f and Path(f).exists()])
    else:
        build_sft(a.clones, a.out or "fim_sft_v03.jsonl", a.per_repo, verified,
                  a.generated if Path(a.generated).exists() else "", a.n_generated)
