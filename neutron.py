"""NeutronAI CLI (`neutron.py`) — Unified Lab Command Center + 14-Check Parallel Suite (Hadron & Quark).

Commands:
  uv run neutron.py doctor      # Run all 14 subsystem self-checks in parallel + inspect toolchains
  uv run neutron.py autopilot . # 1-command end-to-end: Sham-Audit -> Mine -> Decontaminate -> Eval -> Release
  uv run neutron.py eval ...    # Benchmark Hadron / Quark / base models (26 Py/JS/TS/C++/SQL/Perf tasks)
  uv run neutron.py fim ...     # Mine & evaluate single-file + cross-file `<|file_sep|>` Repo-FIM (Quark)
  uv run neutron.py repomap ... # Extract cross-file import graph + compact AST API skeleton map
  uv run neutron.py serve ...   # Start Quark-1.5B -> Hadron-7B Cascade + Auto-Doctest + SQLite stats (:8080/v1)
  uv run neutron.py mcp ...     # Start stdio JSON-RPC 2.0 MCP server (5 tools) for OpenCode / Cursor / Claude
  uv run neutron.py agent ...   # Autonomous shadow-workspace Hadron repo bug-fixer with 1-hop import localization
  uv run neutron.py fuzz ...    # Audit & harden unit tests against RL reward-hacking sham functions
  uv run neutron.py vultr ...   # List Vultr GPU $/hr plans or generate/launch self-destructing GPU jobs
  uv run neutron.py audit ...   # Decontaminate & deduplicate training JSONLs against eval benchmarks
  uv run neutron.py release ... # Generate Hugging Face Model Card + Leaderboard Delta + Ollama Modelfile
  uv run neutron.py mine-ast .. # Mine polyglot bug-fix tasks (Python AST + JS/TS/C++ mutations)
  uv run neutron.py mine-git .. # Mine real bug-fix tasks from any local Git commit history
  uv run neutron.py synth ...   # Rejection-sample verified <think> SFT + Goldilocks RL curriculum
  uv run neutron.py repair ...  # Multi-turn self-repair flywheel + hard-negative DPO harvester
  uv run neutron.py train ...   # Run Unsloth SFT / DPO / Dr.GRPO (Hadron) / Repo-FIM-GRPO (Quark) post-training
"""
import concurrent.futures, os, shutil, subprocess, sys, time

SUBCOMMANDS = {
    "autopilot": "pipeline.py",
    "eval": "eval.py",
    "fim": "fim.py",
    "repomap": "repomap.py",
    "serve": "serve.py",
    "mcp": "mcp_server.py",
    "agent": "agent.py",
    "fuzz": "fuzz_tests.py",
    "vultr": "vultr.py",
    "audit": "audit.py",
    "release": "release.py",
    "mine-ast": "mutate.py",
    "mine-git": "git_mine.py",
    "synth": "synthesize.py",
    "repair": "repair.py",
    "train": "train.py",
}

DOCTOR_CHECKS = [
    ("sandbox (5-lang + algo_ms speed gate)", [sys.executable, "sandbox.py"]),
    ("repomap (import graph + AST skeletons)", [sys.executable, "repomap.py", "--self-check"]),
    ("fim (Quark <|file_sep|> Repo-FIM)", [sys.executable, "fim.py", "--self-check"]),
    ("fuzz_tests (anti-sham test audit)", [sys.executable, "fuzz_tests.py", "--self-check"]),
    ("serve (Quark->Hadron cascade + SQLite)", [sys.executable, "serve.py", "--self-check"]),
    ("mcp (JSON-RPC 2.0 5-tool IDE server)", [sys.executable, "mcp_server.py", "--self-check"]),
    ("agent (Hadron import-graph bug-fixer)", [sys.executable, "agent.py", "--self-check"]),
    ("vultr (auto-destroy GPU cloud-init)", [sys.executable, "vultr.py", "--self-check"]),
    ("audit (benchmark decontamination)", [sys.executable, "audit.py", "--self-check"]),
    ("release (Hadron HF card + Modelfile)", [sys.executable, "release.py", "--self-check"]),
    ("mutate (polyglot Py/TS/C++ bug miner)", [sys.executable, "mutate.py", "--self-check"]),
    ("git_mine (Git commit harvester)", [sys.executable, "git_mine.py", "--self-check"]),
    ("eval (26-task 5-lang + perf pass@k)", [sys.executable, "eval.py", "--mock", "--n", "2", "--k", "1,2"]),
    ("autopilot (full e2e NeutronAI loop)", [sys.executable, "pipeline.py", "--mock"]),
]


def run_doctor() -> int:
    print("=== NeutronAI v0.9 System Check (Hadron & Quark) ===")
    print(f"Python : {sys.version.split()[0]} ({sys.executable})")
    print(
        f"Node26 : {'OK' if shutil.which('node') else 'MISS'} | "
        f"G++23  : {'OK' if shutil.which('g++') else 'MISS'} | "
        f"SQLite3: OK | "
        f"Git    : {'OK' if shutil.which('git') else 'MISS'} | "
        f"Threads: {os.cpu_count()}"
    )
    t0 = time.perf_counter()

    def _run(item):
        label, cmd = item
        s0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        return label, p.returncode == 0, round(time.perf_counter() - s0, 2), (p.stderr or p.stdout).strip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 4)) as pool:
        results = list(pool.map(_run, DOCTOR_CHECKS))

    failed = 0
    for label, ok, sec, out in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label:38s} ({sec:.2f}s)")
        if not ok:
            failed += 1
            print(f"         -> {out.splitlines()[-1] if out else 'unknown error'}")

    total_s = round(time.perf_counter() - t0, 2)
    print(f"\nResult: {len(results) - failed}/{len(results)} checks passed in {total_s}s.")
    return 1 if failed else 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__.strip())
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "doctor":
        sys.exit(run_doctor())
    if cmd not in SUBCOMMANDS:
        print(f"Unknown command `{cmd}`. Valid: doctor, {', '.join(SUBCOMMANDS)}")
        sys.exit(1)
    proc = subprocess.run([sys.executable, SUBCOMMANDS[cmd], *sys.argv[2:]])
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
