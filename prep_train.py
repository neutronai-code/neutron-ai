"""Build the RL training file and a held-out test set from the verified cases.

  uv run prep_train.py --synth-log synth_log.jsonl

* Held-out set: a fixed ~7% of cases chosen by hashing the case id, so the split never changes as cases are added
  and a case can never drift from test into training. Training never sees these; benchmark Hadron on them with
  `uv run eval.py --data tasks_holdout.jsonl ...` to show real improvement, not memorisation.
* Training set: everything else, minus cases the base model always solved (no learning signal), ordered with the
  most informative first: measured ~50% pass rate, then unmeasured cases (assumed hard), then never-solved.
  Measured rates come from synthesize.py's log (`synth_log.jsonl` from the HF dataset repo).
"""
import argparse, collections, hashlib, json, os


def load_jsonl(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_holdout(task_id: str, percent: float) -> bool:
    return int(hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:8], 16) % 10000 < percent * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks_generated.jsonl")
    ap.add_argument("--synth-log", default="synth_log.jsonl", help="synthesize.py per-task log (pass rates)")
    ap.add_argument("--holdout-percent", type=float, default=7.0)
    ap.add_argument("--train-out", default="train_rl.jsonl")
    ap.add_argument("--holdout-out", default="tasks_holdout.jsonl")
    ap.add_argument("--keep-always", action="store_true", help="also train on cases the base model always solved")
    ap.add_argument("--unscored-out", default="",
                    help="Only write the non-held-out cases with no pass rate yet (input for another filter run), then exit")
    ap.add_argument("--source-filter", default="", help="With --unscored-out: only cases whose source contains this")
    args = ap.parse_args()

    tasks = load_jsonl(args.tasks)
    rates = {r["key"]: r for r in load_jsonl(args.synth_log)}
    if not rates:
        print(f"Note: no pass rates found in {args.synth_log}; every case is treated as unmeasured.")
    if args.unscored_out:
        todo = [t for t in tasks if not is_holdout(t["id"], args.holdout_percent) and t["id"] not in rates
                and args.source_filter in t.get("source", "")]
        with open(args.unscored_out, "w", encoding="utf-8") as f:
            for t in todo:
                f.write(json.dumps(t) + "\n")
        langs = collections.Counter(t.get("kind", t.get("lang")) for t in todo)
        print(f"Wrote {len(todo)} unscored cases -> {args.unscored_out}: "
              + ", ".join(f"{k} {v}" for k, v in sorted(langs.items())))
        return

    holdout, train, dropped_always = [], [], 0
    groups = collections.Counter()
    for t in tasks:
        if is_holdout(t["id"], args.holdout_percent):
            holdout.append(t)
            continue
        r = rates.get(t["id"])
        if r is not None:
            if r["n"] >= 4 and r["pass_rate"] == 1.0 and not args.keep_always:
                dropped_always += 1
                continue
            t = {**t, "baseline_pass_rate": r["pass_rate"]}
            groups["never" if r["pass_rate"] == 0 else "sometimes" if r["pass_rate"] < 1 else "always (kept)"] += 1
        else:
            groups["unmeasured"] += 1
        train.append(t)

    # Same order train.py uses (it re-sorts stably with this key), so the file reads in training order.
    train.sort(key=lambda t: 0.3 if t.get("baseline_pass_rate") is None else abs(t["baseline_pass_rate"] - 0.5))
    for path, rows in ((args.train_out, train), (args.holdout_out, holdout)):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    print(f"{len(tasks)} cases -> {len(train)} training ({args.train_out}) + {len(holdout)} held out ({args.holdout_out})")
    print(f"  dropped {dropped_always} that the base model always solved")
    for name in ("sometimes", "unmeasured", "never", "always (kept)"):
        if groups[name]:
            print(f"  {name:14s} {groups[name]}")
    langs = collections.Counter(t.get("kind", t.get("lang")) for t in holdout)
    print("  held out by language: " + ", ".join(f"{k} {v}" for k, v in sorted(langs.items())))
    print(f"\nOne GRPO step trains on one case, so a full pass is --steps {len(train)}.")


if __name__ == "__main__":
    main()
