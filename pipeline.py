"""NeutronAI End-to-End Autopilot (`pipeline.py`) — 1-Command Data -> Audit -> Flywheel -> Eval -> Hadron Release!

Why this completes the NeutronAI lab loop:
  - Orchestrates the **entire $0 local data-and-benchmark preparation run** for **Hadron** & **Quark** in ONE command:
    1. Audits `tasks.jsonl` test suites against reward-hacking shams (`fuzz_tests.py`).
    2. Harvests SFT `<think>` traces + Goldilocks RL curriculum (`synthesize.py`) & hard-negative DPO pairs (`repair.py`) + Quark FIM spans (`fim.py`).
    3. Decontaminates & deduplicates mined training sets against held-out eval tasks (`audit.py`).
    4. Runs parallel 5-language + speed-gate `pass@k` evaluation (`eval.py`).
    5. Emits Hugging Face Model Card + Ollama `Modelfile` (`release.py`) AND self-destructing `vultr_job.sh` (`vultr.py`).

Usage:
  uv run pipeline.py --mock
  uv run pipeline.py --url http://localhost:11434/v1/chat/completions --model hadron-7b --hf-repo NeutronAI/Hadron-7B-v0.9
"""
import argparse, json, os, time
from audit import decontaminate
from eval import MOCK_SOLUTIONS, run_eval
from fim import mine_fim_tasks_from_python
from fuzz_tests import audit_test_strength
from mutate import mine_bug_tasks
from release import build_release_bundle
from repair import run_flywheel
from synthesize import run_engine
from vultr import build_self_destruct_script


def run_autopilot(
    eval_path: str,
    out_dir: str,
    url: str,
    model: str,
    api_key: str,
    hf_repo: str,
    mock: bool,
    repo_url: str = "",
) -> dict:
    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)

    with open(eval_path, encoding="utf-8") as f:
        eval_tasks = [json.loads(l) for l in f if l.strip()]

    # Step 1: Anti-cheat test-suite audit on benchmark
    sham_checks = [audit_test_strength(t) for t in eval_tasks]
    cheatable = [c["id"] for c in sham_checks if not c["airtight"]]
    if cheatable:
        raise RuntimeError(f"Benchmark has reward-hackable test suites: {cheatable}")

    # Step 2: Mine synthetic AST bug-fix tasks + FIM spans (disjoint from eval benchmark!)
    seed_src = (
        "def sliding_max_sum(nums: list[int], k: int) -> int:\n"
        "    if k <= 0 or k > len(nums):\n"
        "        return 0\n"
        "    cur = sum(nums[:k])\n"
        "    best = cur\n"
        "    for i in range(k, len(nums)):\n"
        "        cur = cur + nums[i] - nums[i - k]\n"
        "        if cur > best:\n"
        "            best = cur\n"
        "    return best\n"
    )
    seed_test = (
        "from module import sliding_max_sum\n"
        "assert sliding_max_sum([2, 1, 5, 1, 3, 2], 3) == 9\n"
        "assert sliding_max_sum([4, 2, 1], 5) == 0\n"
        "assert sliding_max_sum([10, -2, 10], 2) == 8\n"
    )
    mined_mutants = mine_bug_tasks("module.py", seed_src, seed_test, max_tasks=4)
    fim_spans = mine_fim_tasks_from_python("sliding_window", seed_src, "assert sliding_max_sum([2,1,5,1,3,2], 3) == 9\nassert sliding_max_sum([1,2], 5) == 0", max_spans=3)

    # Step 3: Run SFT + RL curriculum synthesis + DPO self-repair flywheel
    sft_raw = os.path.join(out_dir, "sft_raw.jsonl")
    rl_raw = os.path.join(out_dir, "rl_raw.jsonl")
    dpo_out = os.path.join(out_dir, "dpo_repairs.jsonl")
    run_engine(eval_path, sft_raw, rl_raw, url, model, api_key, samples=2, mock=mock)
    run_flywheel(eval_path, dpo_out, sft_raw, url, model, api_key, max_turns=2, mock=mock)

    # Step 4: Decontaminate `mined_mutants` against `eval_tasks` to prove clean train/eval separation!
    clean_rl, audit_stats = decontaminate(mined_mutants + eval_tasks[:2], eval_tasks, threshold=0.5)
    clean_rl_path = os.path.join(out_dir, "rl_clean.jsonl")
    fim_path = os.path.join(out_dir, "fim_tasks.jsonl")
    with open(clean_rl_path, "w", encoding="utf-8") as f:
        for r in clean_rl:
            f.write(json.dumps(r) + "\n")
    with open(fim_path, "w", encoding="utf-8") as f:
        for r in fim_spans:
            f.write(json.dumps(r) + "\n")

    # Step 5: Benchmark evaluation
    report_path = os.path.join(out_dir, "eval_report.jsonl")
    eval_summary = run_eval(eval_path, url, model, api_key, n=1, ks=[1], repair_turns=1, workers=6, report_out=report_path, compare_path="", mock=mock)

    # Step 6: Package Hugging Face README + Ollama Modelfile + Self-Destructing Vultr Job Script
    with open(report_path, encoding="utf-8") as f:
        eval_rows = [json.loads(l) for l in f if l.strip()]
    readme_md, modelfile = build_release_bundle(
        model_name="Hadron-7B-v0.9",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        gguf_path="./hadron-7b-v0.9-q4_k_m.gguf",
        eval_rows=eval_rows,
        audit_stats=audit_stats,
    )
    vultr_sh = build_self_destruct_script(
        repo_url=repo_url,  # empty -> the generated script exits immediately instead of training nothing
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        dataset="rl_clean.jsonl",
        steps=300,
        max_minutes=60,
        hf_repo=hf_repo,
    )
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_md)
    with open(os.path.join(out_dir, "Modelfile"), "w", encoding="utf-8") as f:
        f.write(modelfile)
    with open(os.path.join(out_dir, "vultr_job.sh"), "w", encoding="utf-8", newline="\n") as f:
        f.write(vultr_sh)

    manifest = {
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "eval_pass@1": eval_summary["pass@1"],
        "airtight_eval_tasks": len(eval_tasks),
        "clean_rl_tasks": len(clean_rl),
        "leaks_blocked": audit_stats["contaminated_removed"],
        "fim_tasks": len(fim_spans),
        "artifacts_dir": out_dir,
    }
    print(f"\n=== NeutronAI v0.9 Autopilot Complete ===\n{json.dumps(manifest, indent=2)}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tasks.jsonl")
    ap.add_argument("--out-dir", default="build_v0.9")
    ap.add_argument("--url", default="http://localhost:11434/v1/chat/completions")
    ap.add_argument("--model", default="hadron-7b")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--hf-repo", default="NeutronAI/Hadron-7B-v0.9")
    ap.add_argument("--repo-url", default="", help="Git URL the generated vultr_job.sh clones on the VM")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    res = run_autopilot(args.data, args.out_dir, args.url, args.model, args.api_key, args.hf_repo, args.mock, args.repo_url)
    if args.mock:
        assert res["eval_pass@1"] == 1.0 and res["leaks_blocked"] == 2 and res["clean_rl_tasks"] >= 2, res
