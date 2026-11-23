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


def paths(ctx: str, short: str) -> dict[str, str]:
    """Where a prompt set's files live on quark-data. "pilot" = the self-training pilot (molab_sample.py): a greedy
    fill plus N samples per hole, from repos Quark was never trained on."""
    if ctx == "pilot":
        return {"holes": "pilot/holes_ctx2.jsonl", "fills": f"pilot/samples_{short}.json",
                "progress": f"pilot/progress_{short}.jsonl", "report": f"pilot/report_{short}.jsonl"}
    return {"holes": f"bench/repo_holes_{ctx}.jsonl", "fills": f"bench/fills_{short}_{ctx}.json",
            "progress": f"bench/progress_{short}.jsonl", "report": f"bench/report_{ctx}.jsonl"}


def distinct_fills(name: str, samples: list[list[str]]) -> dict[str, list[str]]:
    """Pilot: each hole's greedy + sampled fills -> its distinct fills (same_code: identical up to trailing spaces
    and blank edges), as "models" <name>#0, #1, ... so local_repo_eval tests each distinct fill once. Holes with
    fewer distinct fills get "" there (an empty fill: counted as failed, never tested, ignored by the analysis)."""
    import repo_fim

    per_hole = []
    for fills in samples:
        uniq = []
        for f in fills:
            if f.strip() and not any(repo_fim.same_code(f, u) for u in uniq):
                uniq.append(f)
        per_hole.append(uniq)
    k = max(len(u) for u in per_hole)
    return {f"{name}#{j}": [u[j] if j < len(u) else "" for u in per_hole] for j in range(k)}


def prep(a) -> list[str]:
    """--model may list several models, comma-separated (not for the pilot): all of them are tested in one run."""
    wanted = [m.strip() for m in a.model.split(",") if m.strip()]
    if a.ctx == "pilot" and len(wanted) != 1:
        raise SystemExit("The pilot tests one model at a time")
    names = [version(HfApi(), m) for m in wanted]
    os.makedirs(a.work, exist_ok=True)
    holes_name = paths(a.ctx, "")["holes"]
    holes = get(holes_name, a.work, "holes.jsonl")
    if not holes:
        raise SystemExit(f"quark-data has no {holes_name}")
    rows = [json.loads(line) for line in open(holes, encoding="utf-8") if line.strip()]
    models, progress_lines = {}, []
    for model, name in zip(wanted, names):
        p = paths(a.ctx, name.split("/")[-1])
        saved = get(p["fills"], a.work, "fills_saved.json")
        if not saved:
            maker = "molab_sample.py" if a.ctx == "pilot" else f"molab_fills.py with \"{a.ctx}\" in JOBS"
            raise SystemExit(f"No fills uploaded yet for {name} ({p['fills']}): make them on molab's GPU first "
                             f"({maker} with \"{model}\" in MODELS)")
        data = json.load(open(saved, encoding="utf-8"))
        if a.ctx == "pilot":
            if data["key"]["ids"] != [r["id"] for r in rows]:
                raise SystemExit(f"{p['fills']} was made for other holes than {p['holes']}")
            models.update(distinct_fills(name, data["fills"]))
        else:
            models[name] = data["fills"][name]
        old = get(p["progress"], a.work, "progress_saved.jsonl")
        if old:  # each model's results so far (rows are keyed by model, so they can share one file here)
            progress_lines += [line for line in open(old, encoding="utf-8") if line.strip()]
    with open(os.path.join(a.work, "fills.json"), "w", encoding="utf-8") as f:
        json.dump({"key": {"data": holes_name, "ids": [r["id"] for r in rows]}, "fills": models}, f)
    get("bench/quark_repos.jsonl", a.work, "quark_repos.jsonl")
    with open(os.path.join(a.work, "progress.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(line if line.endswith("\n") else line + "\n" for line in progress_lines)
    n_fills = sum(bool(x.strip()) for fills in models.values() for x in fills)
    print(f"Testing {', '.join(names)} ({a.ctx}: {len(rows)} holes, {n_fills} distinct fills)", flush=True)
    return names


def final(a) -> None:
    names = prep(a)
    label = ", ".join(names) if len(names) < 3 else f"{len(names)} models"
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
    def save_progress():  # each model's rows back to its own progress file on quark-data
        rows = [json.loads(line) for line in open(progress, encoding="utf-8") if line.strip()]
        for name in names:
            mine = [x for x in rows if x["model"] == name or x["model"].startswith(name + "#")]
            HfApi().upload_file(path_or_fileobj="".join(json.dumps(x) + "\n" for x in mine).encode("utf-8"),
                                path_in_repo=paths(a.ctx, name.split("/")[-1])["progress"], repo_id=DATASET,
                                repo_type="dataset", commit_message=f"benchmark progress (GitHub Actions): {name}")

    save_progress()
    p = paths(a.ctx, names[0].split("/")[-1])
    report_name = p["report"]
    report = get(report_name, a.work, "report.jsonl") or os.path.join(a.work, "report.jsonl")
    # Everything the repo jobs tested is in the progress now, so this only tests what is left, then makes the tables.
    # One model: its hub progress file also gets each batch as it finishes. Several: they're saved at the end.
    hub = ["--hub-progress", f"{DATASET}:{p['progress']}"] if len(names) == 1 else []
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "local_repo_eval.py"),
                        "--data", os.path.join(a.work, "holes.jsonl"), "--fills", os.path.join(a.work, "fills.json"),
                        "--report", report, "--repo-info", os.path.join(a.work, "quark_repos.jsonl"),
                        "--workers", "4", "--mem-gb", "16", "--show", "4", "--progress", progress, *hub])
    if len(names) > 1:
        save_progress()
    if r.returncode:
        raise SystemExit("the tests crashed")
    # Runs for other models may have updated the shared report meanwhile (they run at the same time): take the newest
    # copy and replace only this run's models' rows in it.
    ours = lambda x: any(x["model"] == n or x["model"].startswith(n + "#") for n in names)
    mine = [x for x in (json.loads(line) for line in open(report, encoding="utf-8") if line.strip()) if ours(x)]
    latest = get(report_name, a.work, "report_latest.jsonl")
    others = [x for x in (json.loads(line) for line in open(latest, encoding="utf-8") if line.strip())
              if not ours(x)] if latest else []
    HfApi().upload_file(path_or_fileobj="".join(json.dumps(x) + "\n" for x in others + mine).encode("utf-8"),
                        path_in_repo=report_name, repo_id=DATASET, repo_type="dataset",
                        commit_message=f"benchmark results (GitHub Actions): {label}")
    print(f"\nResults uploaded -> {DATASET}/{report_name}. Done!")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prep", "final"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", default="ctx2", choices=["ctx2", "ctx3", "light", "fastlight", "pilot"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--parts", default="", help="final: folder with the repo jobs' progress.jsonl files")
    a = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set: add it as a repository secret (Settings > Secrets and variables > "
                         "Actions > New repository secret, name HF_TOKEN)")
    prep(a) if a.step == "prep" else final(a)


if __name__ == "__main__":
    main()
