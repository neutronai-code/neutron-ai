"""The real-repo FIM benchmark's tests on GitHub Actions (.github/workflows/bench.yml): free for this public repo,
real Linux VMs (molab's notebook sandbox kept being reclaimed mid-run). Fills are made on molab's GPU first
(molab_bench.py uploads them to the quark-data dataset); this only tests them.

The workflow runs one job per benchmark repo, all at once, then a report job:

  python ci_bench.py prep  --model M --ctx ctx2 --work W    holes, fills, repo info and the results so far -> W
  (each repo job then runs local_repo_eval.py --repos <repo> --progress W/progress.jsonl, uploaded as an artifact)
  python ci_bench.py final --model M --ctx ctx2 --work W --parts DIR
      merges the repo jobs' results into quark-data's progress file, tests anything still untested (a repo job that
      failed), prints the tables and merges the results into quark-data/bench/report_<ctx>.jsonl

Results are kept per model VERSION, as in molab_bench.py: "<model>@<sha7>" of the newest commit that changed the
model's own files (a training run's last-checkpoint/ commits don't count). Needs HF_TOKEN.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

DATASET = "neutronai-code/quark-data"
HERE = os.path.dirname(os.path.abspath(__file__))


def version(api: HfApi, model: str) -> str:
    files = [f for f in api.list_repo_tree(model, expand=True)
             if isinstance(f, RepoFile) and f.last_commit and f.path != ".gitattributes" and not f.path.endswith(".md")]
    sha = (max(files, key=lambda f: f.last_commit.date).last_commit.oid if files else api.model_info(model).sha)
    return f"{model}@{sha[:7]}"


def get(name: str, work: str, to: str) -> str | None:
    """quark-data's `name` copied to work/to (None if it isn't there)."""
    try:
        got = hf_hub_download(DATASET, name, repo_type="dataset", local_dir=os.path.join(work, "dl"),
                              force_download=True)
    except Exception as e:
        if "404" in str(e) or "EntryNotFound" in type(e).__name__:
            return None
        raise
    shutil.copyfile(got, os.path.join(work, to))
    return os.path.join(work, to)


def prep(a) -> str:
    name = version(HfApi(), a.model)
    short = name.split("/")[-1]
    os.makedirs(a.work, exist_ok=True)
    holes = get(f"bench/repo_holes_{a.ctx}.jsonl", a.work, "holes.jsonl")
    if not holes:
        raise SystemExit(f"quark-data has no bench/repo_holes_{a.ctx}.jsonl")
    saved = get(f"bench/fills_{short}_{a.ctx}.json", a.work, "fills_saved.json")
    if not saved:
        raise SystemExit(f"No fills uploaded yet for {name} with the {a.ctx} prompts: make them on molab's GPU "
                         f"first (molab_bench.py with MODEL = \"{a.model}\", CTX = \"{a.ctx}\")")
    rows = [json.loads(line) for line in open(holes, encoding="utf-8") if line.strip()]
    fills = json.load(open(saved, encoding="utf-8"))["fills"][name]
    with open(os.path.join(a.work, "fills.json"), "w", encoding="utf-8") as f:
        json.dump({"key": {"data": f"repo_holes_{a.ctx}.jsonl", "ids": [r["id"] for r in rows]},
                   "fills": {name: fills}}, f)
    get("bench/quark_repos.jsonl", a.work, "quark_repos.jsonl")
    if not get(f"bench/progress_{short}.jsonl", a.work, "progress.jsonl"):
        open(os.path.join(a.work, "progress.jsonl"), "w").close()
    print(f"Benchmarking {name} ({a.ctx} prompts, {len(rows)} holes)", flush=True)
    return name


def final(a) -> None:
    name = prep(a)
    short = name.split("/")[-1]
    progress = os.path.join(a.work, "progress.jsonl")
    merged = {}  # (model, id, fill) -> row: a tested result wins over "couldn't test"
    files = [progress] + sorted(glob.glob(os.path.join(a.parts, "**", "progress.jsonl"), recursive=True))
    for path in files:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                x = json.loads(line)
                k = (x["model"], x["id"], x["fill"])
                if k not in merged or merged[k]["pass"] is None:
                    merged[k] = x
    with open(progress, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in merged.values())
    print(f"{len(files) - 1} repo job(s) reported; {sum(x['pass'] is not None for x in merged.values())} test "
          f"results in all", flush=True)
    hub_progress = f"bench/progress_{short}.jsonl"
    HfApi().upload_file(path_or_fileobj=progress, path_in_repo=hub_progress, repo_id=DATASET, repo_type="dataset",
                        commit_message=f"benchmark progress (GitHub Actions): {name}")
    report_name = f"bench/report_{a.ctx}.jsonl"
    report = get(report_name, a.work, "report.jsonl") or os.path.join(a.work, "report.jsonl")
    # Everything the repo jobs tested is in the progress now, so this only tests what is left, then makes the tables.
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "local_repo_eval.py"),
                        "--data", os.path.join(a.work, "holes.jsonl"), "--fills", os.path.join(a.work, "fills.json"),
                        "--report", report, "--repo-info", os.path.join(a.work, "quark_repos.jsonl"),
                        "--workers", "4", "--mem-gb", "16", "--show", "4", "--progress", progress,
                        "--hub-progress", f"{DATASET}:{hub_progress}"])
    if r.returncode:
        raise SystemExit("the tests crashed")
    HfApi().upload_file(path_or_fileobj=report, path_in_repo=report_name, repo_id=DATASET, repo_type="dataset",
                        commit_message=f"benchmark results (GitHub Actions): {name}")
    print(f"\nResults uploaded -> {DATASET}/{report_name}. Done!")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prep", "final"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", default="ctx2", choices=["ctx2", "ctx3"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--parts", default="", help="final: folder with the repo jobs' progress.jsonl files")
    a = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set: add it as a repository secret (Settings > Secrets and variables > "
                         "Actions > New repository secret, name HF_TOKEN)")
    prep(a) if a.step == "prep" else final(a)


if __name__ == "__main__":
    main()
