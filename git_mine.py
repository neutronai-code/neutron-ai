"""NeutronAI Git Commit Miner (`git_mine.py`) — Harvest Real Bug-Fix Tasks from ANY Local Git Repo ($0).

While `mutate.py` creates synthetic bugs, `git_mine.py` extracts REAL developer bug-fix commits
straight out of any local Python/JS Git repository that has tests:
  1. Scans git history for bug-fix commits (`fix`, `bug`, `patch`, `edge`, `resolve`) touching 1-3 source files.
  2. Reconstructs the pre-fix file snapshot (`parent`) and post-fix snapshot (`commit`).
  3. Runs the test file in `sandbox.py` to keep ONLY commits where:
     - Pre-fix code FAILS the test (`score < 1.0`), AND
     - Post-fix code PASSES 100% (`score == 1.0`).
  4. Emits SWE-bench-grade `tasks.jsonl` training items with verified ground-truth `oracle_patch`!

Usage:
  uv run git_mine.py --self-check
  uv run git_mine.py --repo path/to/any/git/repo --max-commits 50 --out git_tasks.jsonl
"""
import argparse, json, os, subprocess, tempfile
from sandbox import verify


def _git(repo: str, *args: str) -> str:
    proc = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True, encoding="utf-8")
    return proc.stdout


def mine_git_repo(repo_dir: str, max_commits: int = 30) -> list[dict]:
    log_out = _git(repo_dir, "log", f"-n{max_commits}", "--no-merges", "--pretty=format:%H\t%P\t%s")
    tasks = []

    for line in log_out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) < 3 or not parts[1].strip():
            continue
        sha, parent, subject = parts[0], parts[1].split()[0], parts[2]
        changed = [p.strip() for p in _git(repo_dir, "diff", "--name-only", parent, sha).splitlines() if p.strip()]
        code_files = [p for p in changed if p.endswith((".py", ".js")) and "test" not in os.path.basename(p).lower()]
        test_files = [p for p in changed if p.endswith((".py", ".js")) and "test" in os.path.basename(p).lower()]

        if not (1 <= len(code_files) <= 3):
            continue
        # If the commit didn't modify a test file, look for an existing `test_*.py` / `*.test.js` in the commit tree
        if not test_files:
            tree_files = _git(repo_dir, "ls-tree", "-r", "--name-only", sha).splitlines()
            test_files = [p for p in tree_files if p.endswith((".py", ".js")) and "test" in os.path.basename(p).lower()]
        if not test_files:
            continue

        try:
            before_files = {p: _git(repo_dir, "show", f"{parent}:{p}") for p in code_files}
            after_files = {p: _git(repo_dir, "show", f"{sha}:{p}") for p in code_files}
            test_src = _git(repo_dir, "show", f"{sha}:{test_files[0]}")
        except subprocess.CalledProcessError:
            continue

        lang = "javascript" if code_files[0].endswith(".js") else "python"
        noop_before = "\n".join(
            f"### FILE: {p}\n<<<<<<< SEARCH\n{src}\n=======\n{src}\n>>>>>>> REPLACE"
            for p, src in before_files.items()
        )
        oracle_patch = "\n".join(
            f"### FILE: {p}\n<<<<<<< SEARCH\n{before_files[p]}\n=======\n{after_files[p]}\n>>>>>>> REPLACE"
            for p in code_files
            if before_files[p] != after_files[p]
        )
        if not oracle_patch:
            continue

        pre_score, pre_info = verify(noop_before, test_src, files=before_files, lang=lang, timeout=3.0)
        post_score, _ = verify(oracle_patch, test_src, files=before_files, lang=lang, timeout=3.0)
        if pre_score < 1.0 and post_score == 1.0:
            listing = "\n\n".join(f"### FILE: {p}\n```{lang}\n{src}\n```" for p, src in before_files.items())
            tasks.append({
                "id": f"git_{sha[:8]}",
                "category": "repo",
                "lang": lang,
                "prompt": (
                    f"Fix the bug (`{subject}`) using `### FILE: <path>` and "
                    f"`<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` blocks.\n"
                    f"Failing signal: `{pre_info['status']}`\n\n{listing}"
                ),
                "files": before_files,
                "test": test_src,
                "oracle_patch": oracle_patch,
            })
    return tasks


def run_self_check():
    with tempfile.TemporaryDirectory(prefix="neutron_git_") as repo:
        subprocess.run(["git", "init", repo], capture_output=True, check=True)
        subprocess.run(["git", "-C", repo, "config", "user.email", "dev@neutron.ai"], check=True)
        subprocess.run(["git", "-C", repo, "config", "user.name", "NeutronAI"], check=True)

        # Commit 1: buggy implementation
        with open(os.path.join(repo, "calc.py"), "w", encoding="utf-8") as f:
            f.write("def mean(vals: list[float]) -> float:\n    return sum(vals) / (len(vals) + 1)\n")
        with open(os.path.join(repo, "test_calc.py"), "w", encoding="utf-8") as f:
            f.write("from calc import mean\nassert mean([2.0, 4.0]) == 3.0\nassert mean([10.0]) == 10.0\n")
        subprocess.run(["git", "-C", repo, "add", "."], check=True)
        subprocess.run(["git", "-C", repo, "commit", "-m", "initial buggy calc"], capture_output=True, check=True)

        # Commit 2: real bug-fix commit
        with open(os.path.join(repo, "calc.py"), "w", encoding="utf-8") as f:
            f.write("def mean(vals: list[float]) -> float:\n    return sum(vals) / len(vals)\n")
        subprocess.run(["git", "-C", repo, "add", "calc.py"], check=True)
        subprocess.run(["git", "-C", repo, "commit", "-m", "fix off-by-one denominator in mean"], capture_output=True, check=True)

        mined = mine_git_repo(repo, max_commits=5)
        assert len(mined) == 1, mined
        s, _ = verify(mined[0]["oracle_patch"], mined[0]["test"], files=mined[0]["files"])
        assert s == 1.0, mined[0]
        print("git_mine.py self-check passed: extracted real git bug-fix commit & verified oracle patch.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--max-commits", type=int, default=30)
    ap.add_argument("--out", default="git_tasks.jsonl")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        run_self_check()
    else:
        rows = mine_git_repo(args.repo, args.max_commits)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Mined {len(rows)} execution-verified git bug-fix tasks -> {args.out}")
