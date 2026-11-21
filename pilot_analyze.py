"""Self-training pilot, the verdict: is there anything for Quark to learn from its own tested fills?

Reads quark-data's pilot/samples_<model>@<sha7>.json (molab_sample.py: a greedy fill + N samples per hole) and
pilot/report_<model>@<sha7>.jsonl (the "Quark pilot tests" workflow: each distinct fill tested), and prints:

  pass@1      the greedy fill passes (what the benchmark measures)
  sample@1    an average sampled fill passes
  pass@all    the greedy fill or any sample passes: the ceiling that training on its own fills could aim for
  learnable   holes where greedy fails but some sample passes: the room to improve
  pairs       holes with at least one passing and one failing fill: preference pairs for DPO

  uv run --no-project --with huggingface_hub python pilot_analyze.py --model neutronai-code/Quark-12B-A2.5B-v0.7
(needs HF_TOKEN; --samples / --report to read local files instead)
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import repo_fim  # noqa: E402


def results(samples: dict, report: list[dict], holes: list[dict] | None = None) -> list[dict]:
    """Per hole: the pass/fail of the greedy fill and of each sample (an empty fill fails)."""
    name = samples["model"]
    tested = collections.defaultdict(list)  # hole id -> [(fill, passed)] of its distinct fills
    for x in report:
        if x["model"].startswith(name + "#") and x["completion"].strip():
            tested[x["id"]].append((x["completion"], x["pass"]))
    out = []
    for i, (hid, fills) in enumerate(zip(samples["key"]["ids"], samples["fills"])):
        def ok(f):
            if not f.strip():
                return False
            for g, p in tested[hid]:
                if repo_fim.same_code(f, g):
                    return bool(p)
            return None  # not tested (yet)
        res = [ok(f) for f in fills]
        row = {"id": hid, "greedy": res[0], "samples": res[1:]}
        if holes:
            row.update(repo=holes[i]["repo"], lang=holes[i]["lang"], size=holes[i]["size"])
        out.append(row)
    return out


def summary(rows: list[dict], label: str = "all") -> str:
    rows = [r for r in rows if r["greedy"] is not None and None not in r["samples"]]
    if not rows:
        return f"{label:38s}   (nothing tested)"
    n = len(rows)
    p1 = sum(r["greedy"] for r in rows)
    s1 = sum(sum(r["samples"]) / len(r["samples"]) for r in rows)
    best = sum(r["greedy"] or any(r["samples"]) for r in rows)
    learn = sum((not r["greedy"]) and any(r["samples"]) for r in rows)
    pairs = sum(any([r["greedy"], *r["samples"]]) and not all([r["greedy"], *r["samples"]]) for r in rows)
    pct = lambda k: f"{k / n * 100:5.1f}%"
    return (f"{label:38s} {n:5d}  pass@1 {pct(p1)}  sample@1 {pct(s1)}  pass@all {pct(best)}  "
            f"learnable {learn:4d} ({pct(learn)})  pairs {pairs:4d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="neutronai-code/Quark-12B-A2.5B-v0.7")
    ap.add_argument("--samples", default="", help="local samples JSON (default: download from quark-data)")
    ap.add_argument("--report", default="", help="local report JSONL (default: download from quark-data)")
    ap.add_argument("--holes", default="", help="local pilot holes JSONL, for the per-repo / language table")
    a = ap.parse_args()
    if not (a.samples and a.report):
        from huggingface_hub import hf_hub_download

        from ci_bench import DATASET, version
        from huggingface_hub import HfApi

        short = version(HfApi(), a.model).split("/")[-1]
        dl = lambda p: hf_hub_download(DATASET, p, repo_type="dataset", force_download=True)
        a.samples = a.samples or dl(f"pilot/samples_{short}.json")
        a.report = a.report or dl(f"pilot/report_{short}.jsonl")
        a.holes = a.holes or dl("pilot/holes_ctx2.jsonl")
    samples = json.load(open(a.samples, encoding="utf-8"))
    report = [json.loads(line) for line in open(a.report, encoding="utf-8") if line.strip()]
    holes = [json.loads(line) for line in open(a.holes, encoding="utf-8") if line.strip()] if a.holes else None
    rows = results(samples, report, holes)
    untested = sum(r["greedy"] is None or None in r["samples"] for r in rows)
    print(f"{samples['model']}: {len(rows)} holes, greedy + {len(samples['fills'][0]) - 1} samples at temperature "
          f"{samples.get('temperature')}" + (f" ({untested} holes not fully tested, left out)" if untested else ""))
    print(summary(rows))
    if holes:
        for key in ("lang", "size", "repo"):
            print()
            for g in sorted({r[key] for r in rows}):
                print(summary([r for r in rows if r[key] == g], f"  {g}"))


if __name__ == "__main__":
    main()
