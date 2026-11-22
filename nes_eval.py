"""Score next-edit predictions (nes_data.py rows) and query a model for them.

A prediction is the model's rewrite of the marked region. Scores, on the benchmark rows (held-out repos):
  - edit rows: EXACT  = the rewrite equals the real next version of the region (trailing spaces ignored)
               MOVED  = the rewrite changed the region at all (it noticed an edit belongs here)
  - none rows: QUIET  = the rewrite left the region unchanged (it didn't invent an edit)
  - overall:   the share of rows where the rewrite equals the target (edit and none together)
Nothing runs the code: an exact rewrite of a real edit is a strict, cheap measure, as in Zed's Zeta evaluations.

  python nes_eval.py --data nes_bench.jsonl --preds preds.json      # preds: {"<row id>": "<rewrite>"}
  python nes_eval.py --data nes_bench.jsonl --baseline copy         # the do-nothing baseline
"""
import argparse
import collections
import json
import urllib.request

from nes_data import CURSOR, NES_STOPS, REGION_END, REGION_START


def shown_region(row: dict) -> str:
    """The region as the prompt shows it, without the cursor marker."""
    return row["prompt"].split(f"{REGION_START}\n", 1)[1].split(f"\n{REGION_END}", 1)[0].replace(CURSOR, "")


def target_region(row: dict) -> str:
    return row["target"][: -len(f"\n{REGION_END}")]


def norm(text: str) -> str:
    return "\n".join(l.rstrip() for l in text.strip("\n").split("\n"))


def clean(pred: str) -> str:
    """A raw model output -> the rewritten region (cut at the region end marker, if the stop didn't)."""
    return pred.split(REGION_END, 1)[0].rstrip("\n")


def score(rows: list[dict], preds: dict[str, str]) -> dict:
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        p = norm(clean(preds.get(r["id"], "")))
        shown, target = norm(shown_region(r)), norm(target_region(r))
        for key in ("all", f"lang:{r['lang']}"):
            c = by[key]
            c["n"] += 1
            c["right"] += p == target
            if r["kind"] == "edit":
                c["edit"] += 1
                c["exact"] += p == target
                c["moved"] += p != shown
            else:
                c["none"] += 1
                c["quiet"] += p == shown
    out = {}
    for key, c in by.items():
        out[key] = {"rows": c["n"], "overall": 100 * c["right"] / c["n"],
                    "exact_edit": 100 * c["exact"] / max(1, c["edit"]), "moved_edit": 100 * c["moved"] / max(1, c["edit"]),
                    "quiet_none": 100 * c["quiet"] / max(1, c["none"]), "edit_rows": c["edit"], "none_rows": c["none"]}
    return out


def query_nes(url: str, model: str, prompt: str, max_tokens: int = 1024, api_key: str = "") -> str:
    """Raw /v1/completions request (vLLM or llama-server's OpenAI endpoint), greedy, stopping at the region end."""
    payload = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                          "stop": NES_STOPS, "skip_special_tokens": False}).encode("utf-8")
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
    with urllib.request.urlopen(urllib.request.Request(url, payload, headers), timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["text"]


def show(scores: dict) -> None:
    print(f"{'':16} {'rows':>6} {'overall':>8} {'exact (edit)':>13} {'moved (edit)':>13} {'quiet (none)':>13}")
    for key in ["all"] + sorted(k for k in scores if k != "all"):
        s = scores[key]
        print(f"{key:16} {s['rows']:>6} {s['overall']:>7.1f}% {s['exact_edit']:>12.1f}% {s['moved_edit']:>12.1f}% "
              f"{s['quiet_none']:>12.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="nes_bench.jsonl")
    ap.add_argument("--preds", default="")
    ap.add_argument("--baseline", choices=["copy", "oracle"], default="")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.baseline == "copy":
        preds = {r["id"]: shown_region(r) for r in rows}
    elif a.baseline == "oracle":
        preds = {r["id"]: target_region(r) for r in rows}
    else:
        preds = json.load(open(a.preds, encoding="utf-8"))
        preds = preds.get("preds", preds)
    show(score(rows, preds))


if __name__ == "__main__":
    main()
