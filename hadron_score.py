"""Score Hadron bake-off answers: step 2 of molab_hadron_bench.py, run off molab.

molab_hadron_bench.py uploads each model's answers (4 per task) to hadron-data gen/<TAG>/<tasks>.json. This script
runs every answer's tests with sandbox.verify, exactly as eval.py does, writes eval.py-style reports to
eval/<TAG>/<tasks>.jsonl + .txt, and prints one table comparing every scored model, with a paired per-task
comparison against the baseline (Qwen3.5-4B).

  uv run --with huggingface_hub python hadron_score.py              # every model whose answers are complete
  uv run --with huggingface_hub python hadron_score.py Qwen3.5-9B-32k --redo
  uv run --with huggingface_hub python hadron_score.py --no-upload  # score and print only
  uv run --with huggingface_hub python hadron_score.py Mellum2.1-12B-A2.5B-Thinking-32k --set train
      # training tasks: pass rates -> rl_tasks_<tag>.jsonl (RL set) + distill_targets_<tag>.jsonl (never solved)
      # (or on GitHub Actions: Actions > "Hadron task scoring" > Run workflow, one job per part)

Needs node (Node 22.7+) and g++ (C++23) on PATH, as eval.py does. Every model is scored on the same machine, so
the comparison is fair; absolute numbers can differ slightly from a Linux run (process start-up time, compilers).
"""
import argparse, collections, concurrent.futures, json, math, os, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval import pass_at_k  # noqa: E402
from sandbox import verify  # noqa: E402

DATASET = "neutronai-code/hadron-data"
FILES = ["tasks_holdout.jsonl", "tasks.jsonl"]
BASELINE = "Qwen3.5-4B-32k"


def load_tasks(name: str) -> dict:
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return {t["id"]: t for t in map(json.loads, f) if t}


def score_task(task: dict, answers: list) -> dict:
    """eval.py's eval_single_task, on answers made earlier."""
    rewards, statuses, latencies, passed = [], [], [], 0
    for content, _finish, _tokens in answers:
        score, info = verify(content, task["test"], task.get("starter_code", ""), task.get("files"),
                             task.get("lang", "python"), perf_budget_ms=float(task.get("perf_budget_ms", 0.0)))
        rewards.append(score)
        statuses.append(info["status"])
        latencies.append(info.get("ms", 0.0))
        passed += score == 1.0
    n = len(answers)
    return {"id": task.get("id", "unknown"), "category": task.get("category", "general"), "n": n, "c": passed,
            "pass@1": round(pass_at_k(n, passed, 1), 4), "mean_reward": round(sum(rewards) / n, 4),
            "exec_ms": round(sum(latencies) / n, 2), "status": statuses[0],
            "cut_off": sum(a[1] == "length" for a in answers), "empty": sum(not (a[0] or "").strip() for a in answers),
            "tokens": [a[2] for a in answers]}


def summarize(tag: str, data: str, key: dict, rows: list) -> str:
    """The summary eval.py prints, plus how often answers were empty or cut off."""
    out = [f"=== {key['model']}{' step ' + str(key['step']) if key.get('step') else ''} ({key['rev'][:7]}) on {data} "
           f"(answers from molab, {key['ctx'] // 1024}k context, n={key['n']}, temperature={key['temperature']}; "
           f"tests run locally by hadron_score.py) ==="]
    for r in rows:
        out.append(f"[{r['category']:8s}] {r['id']:28s} :: {r['c']}/{r['n']} passed | reward={r['mean_reward']:.2f} | "
                   f"{r['exec_ms']:5.1f}ms ({r['status']})")
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    out.append("\nCategory pass@1:")
    for cat, items in sorted(by_cat.items()):
        out.append(f"  {cat:10s}: {sum(pass_at_k(r['n'], r['c'], 1) for r in items) / len(items):.1%} ({len(items)} tasks)")
    n = rows[0]["n"]
    p1 = sum(pass_at_k(r["n"], r["c"], 1) for r in rows) / len(rows)
    pk = sum(pass_at_k(r["n"], r["c"], n) for r in rows) / len(rows)
    dense = sum(r["mean_reward"] for r in rows) / len(rows)
    total = sum(r["n"] for r in rows)
    toks = sorted(t for r in rows for t in r["tokens"])
    out.append(f"\nOverall: pass@1={p1:.1%} | pass@{n}={pk:.1%} | mean_dense_reward={dense:.3f}")
    out.append(f"Answers: {total}, cut off at the context limit {sum(r['cut_off'] for r in rows)}, "
               f"empty {sum(r.get('empty', 0) for r in rows)} | "
               f"median {toks[len(toks) // 2]} tokens, max {toks[-1]}")
    return "\n".join(out)


def sign_test(wins: int, losses: int) -> float:
    m, k = wins + losses, min(wins, losses)
    return min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0


def score_train(tag: str, args, api, files: list, cache: str) -> None:
    """Training tasks: score each finished part (gen/<tag>/train/part-NNN.json -> eval/<tag>/train/part-NNN.jsonl,
    kept so a part is tested once), then, when every part is in, write the per-task pass rates and two task files:
      rl_tasks_<tag>.jsonl       tasks the model solves sometimes (1-3 of 4): the RL set, with baseline_pass_rate
      distill_targets_<tag>.jsonl tasks it never solves (0 of 4): for teacher distillation (distill.py --data)
    Tasks it always solves teach nothing and are left out. The task files are local (keep them out of git); the
    rates also go to rates/<tag>.jsonl on the Hub."""
    from huggingface_hub import hf_hub_download

    held_out = set(load_tasks("tasks_holdout.jsonl"))
    with open(os.path.join(HERE, "tasks_generated.jsonl"), encoding="utf-8") as f:
        pool = [t for t in map(json.loads, f) if t and t["id"] not in held_out]
    by_id = {t["id"]: t for t in pool}
    parts = sorted(f for f in files if f.startswith(f"gen/{tag}/train/part-") and f.endswith(".json"))
    if args.part:  # one part only (a GitHub Actions job): score and upload it, nothing else
        parts = [f for f in parts if f.endswith(f"/{args.part}.json")]
        if not parts:
            print(f"{tag}: no answers for {args.part}: nothing to do")
            return
    if not parts:
        print(f"{tag}: no training-task answers yet")
        return
    rows, waiting = [], []
    for gen_path in parts:
        name = gen_path[len(f"gen/{tag}/"):-len(".json")]
        report_path = f"eval/{tag}/{name}.jsonl"
        if report_path in files and not args.redo:
            path = hf_hub_download(DATASET, report_path, repo_type="dataset", local_dir=cache, force_download=True)
            rows += [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            continue
        gen = json.load(open(hf_hub_download(DATASET, gen_path, repo_type="dataset", local_dir=cache,
                                             force_download=True), encoding="utf-8"))
        if not gen.get("complete"):
            waiting.append(f"{name} ({len(gen['answers'])} tasks so far)")
            continue
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
            part_rows = list(ex.map(lambda i: score_task(by_id[i], gen["answers"][i]), list(gen["answers"])))
        for r in part_rows:
            r["lang"] = by_id[r["id"]].get("lang", "python")
        print(f"{tag} / {name}: {len(part_rows)} tasks, pass@1 "
              f"{sum(r['c'] / r['n'] for r in part_rows) / len(part_rows):.1%} (tests took {time.time() - t0:.0f}s)")
        if not args.no_upload:
            api.upload_file(path_or_fileobj="".join(json.dumps(r) + "\n" for r in part_rows).encode("utf-8"),
                            path_in_repo=report_path, repo_id=DATASET, repo_type="dataset",
                            commit_message=f"{tag}: scored {name} (hadron_score.py)")
        rows += part_rows
    if args.part:
        return
    print(f"\n{tag}: {len(rows)} of {len(pool)} training tasks scored"
          + (f"; still answering: {', '.join(waiting)}" if waiting else ""))
    if not rows:
        return

    groups = {"always": [], "sometimes": [], "never": []}
    for r in rows:
        groups["always" if r["c"] == r["n"] else "never" if r["c"] == 0 else "sometimes"].append(r)
    langs = sorted({r["lang"] for r in rows})
    print(f"  {'':10s} " + " ".join(f"{l:>12s}" for l in langs) + f" {'all':>7s}")
    for g, rs in groups.items():
        print(f"  {g:10s} " + " ".join(f"{sum(r['lang'] == l for r in rs):12d}" for l in langs) + f" {len(rs):7d}")
    print(f"  pass@1 over these tasks: {sum(r['c'] / r['n'] for r in rows) / len(rows):.1%}")
    if waiting:
        print("  (task files are written once every part is scored)")
        return

    rates = [{"id": r["id"], "lang": r["lang"], "category": r["category"], "n": r["n"], "c": r["c"],
              "rate": r["c"] / r["n"], "cut_off": r.get("cut_off", 0), "tokens": r.get("tokens", [])} for r in rows]
    if not args.no_upload:
        api.upload_file(path_or_fileobj="".join(json.dumps(r) + "\n" for r in rates).encode("utf-8"),
                        path_in_repo=f"rates/{tag}.jsonl", repo_id=DATASET, repo_type="dataset",
                        commit_message=f"{tag}: pass rates on the training tasks")
    for fname, group in ((f"rl_tasks_{tag}.jsonl", "sometimes"), (f"distill_targets_{tag}.jsonl", "never")):
        with open(os.path.join(HERE, fname), "w", encoding="utf-8") as f:
            for r in groups[group]:
                f.write(json.dumps(dict(by_id[r["id"]], baseline_pass_rate=r["c"] / r["n"])) + "\n")
        print(f"wrote {fname}: {len(groups[group])} tasks")
        if not args.no_upload:  # also on the Hub: for the training notebook, and when this runs on GitHub Actions
            api.upload_file(path_or_fileobj=os.path.join(HERE, fname), path_in_repo=f"rates/{fname}", repo_id=DATASET,
                            repo_type="dataset", commit_message=f"{tag}: {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", help="Model tags under gen/ (default: every one with complete answers)")
    ap.add_argument("--redo", action="store_true", help="Score again even if eval/<TAG>/ already has a report")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--part", default="", help="--set train: score only this part (e.g. part-003), as one CI job")
    ap.add_argument("--set", choices=["bench", "train"], default="bench",
                    help="bench: the held-out + hand-made tasks (a score table); train: the training tasks (pass rates, "
                         "the RL task list and the never-solved tasks for distillation)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(DATASET, repo_type="dataset")
    tags = args.tags or sorted({f.split("/")[1] for f in files if f.startswith("gen/") and f.endswith(".json")})
    if not tags:
        print(f"No answers in {DATASET} gen/ yet: run molab_hadron_bench.py first")
        return
    cache = os.path.join(tempfile.gettempdir(), "hadron_score")
    if args.set == "train":
        for tag in tags:
            score_train(tag, args, api, files, cache)
        return
    tasks = {f: load_tasks(f) for f in FILES}
    reports = {}  # tag -> {task id: row} for the held-out set
    for tag in tags:
        for data in FILES:
            stem = data[:-len(".jsonl")]
            if f"gen/{tag}/{stem}.json" not in files:
                print(f"{tag}: no answers for {data} yet")
                continue
            report_path = f"eval/{tag}/{stem}.jsonl"
            if report_path in files and not args.redo:
                path = hf_hub_download(DATASET, report_path, repo_type="dataset", local_dir=cache, force_download=True)
                rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
                print(f"{tag} / {data}: already scored ({report_path}); --redo to score again")
            else:
                gen = json.load(open(hf_hub_download(DATASET, f"gen/{tag}/{stem}.json", repo_type="dataset",
                                                     local_dir=cache, force_download=True), encoding="utf-8"))
                if not gen.get("complete"):
                    print(f"{tag} / {data}: answers not complete yet ({len(gen['answers'])} tasks so far)")
                    continue
                missing = [i for i in tasks[data] if i not in gen["answers"]]
                assert not missing, f"{tag}: no answers for {missing[:5]}"
                t0 = time.time()
                with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
                    rows = list(pool.map(lambda t: score_task(t, gen["answers"][t["id"]]), tasks[data].values()))
                text = summarize(tag, data, gen, rows)
                print(text[text.index("\nCategory"):] if len(text) > 4000 else text)
                print(f"({tag} / {data}: tests took {time.time() - t0:.0f}s)")
                if not args.no_upload:
                    api.upload_file(path_or_fileobj="".join(json.dumps(r) + "\n" for r in rows).encode("utf-8"),
                                    path_in_repo=report_path, repo_id=DATASET, repo_type="dataset",
                                    commit_message=f"{tag}: scored {data} (hadron_score.py)")
                    api.upload_file(path_or_fileobj=text.encode("utf-8"), path_in_repo=f"eval/{tag}/{stem}.txt",
                                    repo_id=DATASET, repo_type="dataset",
                                    commit_message=f"{tag}: summary for {data}")
            if stem == "tasks_holdout":
                reports[tag] = {r["id"]: r for r in rows}

    if not reports:
        return
    p1 = lambda r: r["c"] / r["n"]  # noqa: E731
    print(f"\nHeld-out tasks ({len(tasks['tasks_holdout.jsonl'])}), 4 answers each, 32k context:")
    print(f"  {'model':40s} pass@1  pass@4   vs {BASELINE} (tasks better / worse, sign test)")
    base = reports.get(BASELINE)
    for tag, rep in sorted(reports.items(), key=lambda kv: -sum(map(p1, kv[1].values()))):
        rows = list(rep.values())
        a = sum(map(p1, rows)) / len(rows)
        b = sum(pass_at_k(r["n"], r["c"], r["n"]) for r in rows) / len(rows)
        cmp = ""
        if base and tag != BASELINE:
            both = [i for i in rep if i in base]
            d = [p1(rep[i]) - p1(base[i]) for i in both]
            w, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
            cmp = f"{sum(d) / len(both) * 100:+5.1f} pts ({w} / {l}, p={sign_test(w, l):.3f})"
        print(f"  {tag:40s} {a:6.1%}  {b:6.1%}   {cmp}")


if __name__ == "__main__":
    main()
