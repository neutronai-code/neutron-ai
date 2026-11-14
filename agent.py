"""NeutronAI Hadron Repo Agent (`agent.py`) — Autonomous Local Repository Bug-Fixer & SWE Trajectory Recorder.

Why every serious coding lab (Anthropic Claude Code, Cognition, Cursor) builds their own agent loop:
  - `sandbox.py` trains models on pre-sliced snippets, whereas REAL software engineering starts with:
    *"Tests are failing in `my_project/` — go find the right files, patch them, and make tests pass."*
  - `agent.py` works on ANY real local directory (in an isolated temp workspace copy so it NEVER corrupts your working tree unless `--apply` is passed!):
    1. Runs your test command (`pytest`, `python -m unittest`, `node --test`, etc.) to capture the real failure traceback.
    2. Ranks & localizes the most relevant source files mentioned in the traceback + 1-hop import graph (`repomap.py`).
    3. Generates a `### FILE:` `SEARCH/REPLACE` patch, applies it to the shadow workspace, and re-runs tests up to `--max-turns`.
    4. When tests go RED -> GREEN, optionally writes the verified fix back (`--apply`) AND appends a clean SWE-agent training trajectory to `agent_trajectories.jsonl`!

Usage:
  uv run agent.py --self-check
  uv run agent.py --repo ./some_project --test-cmd "python test_suite.py" --goal "Fix failing edge cases" --apply
"""
import argparse, json, math, os, re, shutil, subprocess, sys, tempfile, urllib.request
from repomap import build_import_graph, build_repo_map
from sandbox import apply_search_replace_blocks

AGENT_SYS = (
    "You are Hadron, an autonomous repository engineering model by NeutronAI.\n"
    "Think step-by-step inside <think>...</think> to diagnose the root cause from the failing test output "
    "and source files, then output ONLY exact `### FILE: <relative_path>` and "
    "`<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` blocks."
)

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "dist", "build"}
_CODE_EXTS = (".py", ".js", ".ts", ".mts", ".cjs", ".mjs", ".cpp", ".cc", ".cxx", ".hpp", ".h", ".sql")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


def collect_repo_files(repo_dir: str, max_file_bytes: int = 32_000) -> dict[str, str]:
    files = {}
    root = os.path.abspath(repo_dir)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(_CODE_EXTS):
                continue
            full = os.path.join(dirpath, fn)
            if os.path.getsize(full) > max_file_bytes:
                continue
            rel = os.path.relpath(full, root).replace("\\", "/")
            try:
                with open(full, encoding="utf-8") as f:
                    files[rel] = f.read()
            except UnicodeDecodeError:
                continue
    return files


def localize_files(files: dict[str, str], query_and_trace: str, top_k: int = 5) -> list[str]:
    """Rank repository files by traceback mentions + IDF overlap + 1-hop import-graph propagation."""
    code_files = {p: c for p, c in files.items() if "test" not in os.path.basename(p).lower()} or files
    q_tokens = set(_WORD_RE.findall(query_and_trace.lower()))
    doc_freq = {}
    tokenized = {}
    for path, content in code_files.items():
        toks = set(_WORD_RE.findall((path + "\n" + content).lower()))
        tokenized[path] = toks
        for tok in toks:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    n_docs = len(code_files) + 1
    base_scores = {}
    for path, toks in tokenized.items():
        trace_hit = 10.0 if (path in query_and_trace or os.path.basename(path) in query_and_trace) else 0.0
        idf_sum = sum(math.log(n_docs / doc_freq[t]) + 1.0 for t in (q_tokens & toks) if t in doc_freq)
        base_scores[path] = trace_hit + idf_sum

    # Propagate 45% of high-signal file score across 1-hop callees & 25% to callers so hidden imported bugs surface!
    graph = build_import_graph(code_files)
    final_scores = dict(base_scores)
    for src, deps in graph.items():
        for dep in deps:
            if dep in final_scores:
                final_scores[dep] += 0.45 * base_scores[src]
                final_scores[src] += 0.25 * base_scores[dep]

    ranked = sorted(final_scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return [p for p, _ in ranked[:top_k]]


def run_test_cmd(cwd: str, test_cmd: list[str], timeout: float = 15.0) -> tuple[bool, str]:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=cwd)
    try:
        proc = subprocess.run(test_cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        return proc.returncode == 0, out[-2500:]
    except subprocess.TimeoutExpired:
        return False, f"TimeoutExpired after {timeout}s"


def call_model(url: str, model: str, messages: list[dict], api_key: str) -> str:
    payload = json.dumps({"model": model, "temperature": 0.2, "messages": messages}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]


def run_repo_agent(
    repo_dir: str,
    test_cmd: list[str],
    goal: str,
    url: str = "",
    model: str = "",
    api_key: str = "",
    max_turns: int = 3,
    apply_changes: bool = False,
    traj_log: str = "agent_trajectories.jsonl",
    mock_fn=None,
) -> dict:
    orig_files = collect_repo_files(repo_dir)
    if not orig_files:
        raise ValueError(f"No supported source files found in {repo_dir}")

    with tempfile.TemporaryDirectory(prefix="neutron_agent_") as shadow:
        shutil.copytree(repo_dir, shadow, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*_SKIP_DIRS))
        ok0, trace0 = run_test_cmd(shadow, test_cmd)
        if ok0:
            return {"solved": True, "turns": 0, "reason": "already_passing"}

        focus_paths = localize_files(orig_files, f"{goal}\n{trace0}", top_k=5)
        other_files = {k: v for k, v in orig_files.items() if k not in focus_paths}
        skeleton_hdr = build_repo_map(other_files, max_chars=2500) if other_files else ""
        context_blob = "\n\n".join(f"### FILE: {p}\n```\n{orig_files[p]}\n```" for p in focus_paths)
        map_section = f"\n\nRepository API Map (other files):\n```\n{skeleton_hdr}\n```" if skeleton_hdr else ""
        user_msg = f"Goal: {goal}\n\nFailing test output:\n```\n{trace0}\n```{map_section}\n\nRelevant repository files:\n{context_blob}"
        messages = [{"role": "system", "content": AGENT_SYS}, {"role": "user", "content": user_msg}]

        cur_files = dict(orig_files)
        for turn in range(1, max_turns + 1):
            reply = mock_fn(turn, focus_paths, messages) if mock_fn else call_model(url, model, messages, api_key)
            patched, p_status = apply_search_replace_blocks(cur_files, reply)
            if patched is None:
                messages.extend([
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": f"Patch failed to apply (`{p_status}`). Output exact matching SEARCH/REPLACE blocks."},
                ])
                continue

            for rel, content in patched.items():
                if cur_files.get(rel) != content:
                    full = os.path.join(shadow, rel)
                    os.makedirs(os.path.dirname(full), exist_ok=True)
                    with open(full, "w", encoding="utf-8") as f:
                        f.write(content)
            cur_files = patched

            ok, trace = run_test_cmd(shadow, test_cmd)
            if ok:
                if apply_changes:
                    for rel, content in cur_files.items():
                        if orig_files.get(rel) != content:
                            with open(os.path.join(repo_dir, rel), "w", encoding="utf-8") as f:
                                f.write(content)
                if traj_log:
                    with open(traj_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "goal": goal,
                            "localized_files": focus_paths,
                            "turns": turn,
                            "messages": [
                                {"role": "system", "content": AGENT_SYS},
                                {"role": "user", "content": user_msg},
                                {"role": "assistant", "content": reply},
                            ],
                        }) + "\n")
                return {"solved": True, "turns": turn, "localized": focus_paths, "patch": reply}

            messages.extend([
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"Tests still failing after patch:\n```\n{trace}\n```\nFix the remaining issue."},
            ])

    return {"solved": False, "turns": max_turns, "localized": focus_paths}


def run_self_check():
    with tempfile.TemporaryDirectory(prefix="neutron_repotest_") as repo:
        os.makedirs(os.path.join(repo, "billing"), exist_ok=True)
        # Distractor files + 1 genuinely buggy file to test automatic file localization!
        with open(os.path.join(repo, "billing", "__init__.py"), "w", encoding="utf-8") as f:
            f.write("")
        with open(os.path.join(repo, "billing", "logger.py"), "w", encoding="utf-8") as f:
            f.write("def info(msg: str) -> str:\n    return f'[INFO] {msg}'\n")
        with open(os.path.join(repo, "billing", "tax.py"), "w", encoding="utf-8") as f:
            f.write("def apply_tax(subtotal: float, rate: float) -> float:\n    return round(subtotal - subtotal * rate, 2)\n")
        with open(os.path.join(repo, "test_billing.py"), "w", encoding="utf-8") as f:
            f.write("from billing.tax import apply_tax\nassert apply_tax(100.0, 0.08) == 108.0\n")

        def fake_model(turn, localized, _msgs):
            assert localized[0] == "billing/tax.py", f"Expected billing/tax.py ranked #1, got {localized}"
            if turn == 1:
                return "### FILE: billing/tax.py\n<<<<<<< SEARCH\nmissing_line\n=======\nx\n>>>>>>> REPLACE"
            return (
                "<think>billing/tax.py subtracts tax instead of adding it.</think>\n"
                "### FILE: billing/tax.py\n"
                "<<<<<<< SEARCH\n"
                "    return round(subtotal - subtotal * rate, 2)\n"
                "=======\n"
                "    return round(subtotal + subtotal * rate, 2)\n"
                ">>>>>>> REPLACE"
            )

        res = run_repo_agent(
            repo,
            [sys.executable, "test_billing.py"],
            goal="Fix tax calculation failure",
            apply_changes=True,
            traj_log="",
            mock_fn=fake_model,
        )
        assert res["solved"] is True and res["turns"] == 2, res
        ok_after, _ = run_test_cmd(repo, [sys.executable, "test_billing.py"])
        assert ok_after is True

        # Verify 1-hop import-graph localization surfaces a buggy dependency NOT mentioned in traceback!
        dep_repo = {
            "pkg/entry.py": "from pkg.hidden_math import scale\ndef compute(x): return scale(x)\n",
            "pkg/hidden_math.py": "def scale(n): return n - 1\n",
            "pkg/distractor1.py": "def log_event(e): return str(e)\n",
            "pkg/distractor2.py": "def fmt_name(s): return s.strip()\n",
        }
        top2 = localize_files(dep_repo, "AssertionError in pkg/entry.py line 2: compute(5)", top_k=2)
        assert top2 == ["pkg/entry.py", "pkg/hidden_math.py"], top2
        print("agent.py self-check passed: localized buggy file #1 + 1-hop hidden import, survived bad patch turn, verified RED->GREEN fix.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--test-cmd", default="python -m unittest")
    ap.add_argument("--goal", default="Fix failing tests")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="hadron-7b")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--apply", action="store_true", help="Write verified fix back to --repo")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        run_self_check()
    else:
        res = run_repo_agent(
            args.repo,
            args.test_cmd.split(),
            args.goal,
            args.url,
            args.model,
            args.api_key,
            args.max_turns,
            args.apply,
        )
        print(json.dumps(res, indent=2))
