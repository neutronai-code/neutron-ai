"""Next-edit prediction ("NES") data for Quark, from real commit histories.

The task (v1, "nearby next edit", the shape of Zed's Zeta): the model sees the user's last few edits (small diffs) and
the file around the cursor with an editable REGION marked (~10-25 lines around the last edit), and writes that region
again with the next edit(s) applied, or unchanged when no edit belongs there. The IDE finds farther targets by asking
about candidate spots (other uses of a renamed name, call sites) and shows the rewrite as an inline diff.

Data: commits of the permissively licensed training repos (quark_data_v04.load_repos), held-out benchmark repos
(repo_fim.HELDOUT) only with --heldout. A commit's diff hunks, in file order, stand in for the order the edits were made:
the first k hunks of a file are the history, the next one the target. Skipped: merges, bots, commits with AI-tool
co-author trailers or "generated with" notes (training on them would be distilling from those models), huge or
formatting-only commits, files added or deleted by the commit, non-source files (tests, generated, minified).

  python nes_data.py --repos 5 --out nes_smoke.jsonl          # smoke test: first 5 training repos
  python nes_data.py --out nes_train.jsonl                    # all training repos
  python nes_data.py --heldout --out nes_bench.jsonl          # the benchmark repos

Each row: {"id", "repo", "commit", "path", "lang", "kind": "edit"|"none", "prompt", "target"}. The target is the new
region followed by REGION_END; train.py adds <|endoftext|>. The clones live in ~/quark_repos (shallow ones are deepened
with `git fetch --unshallow`, which only adds history: HEAD stays at the pinned commit).
"""
import argparse
import collections
import concurrent.futures as cf
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from quark_data_v04 import EXT_LANG, GENERATED, MAX_LINE_CHARS, git, glob_regex, is_source, load_repos
from repo_fim import HELDOUT, REPOS

HISTORY, EXCERPT, REGION_START, REGION_END, CURSOR, REWRITE = (
    "<|edit_history|>", "<|excerpt|>", "<|region_start|>", "<|region_end|>", "<|cursor|>", "<|rewrite|>")
NES_STOPS = [REGION_END, "<|endoftext|>"]

MAX_COMMITS = 600          # newest commits read per repo
MAX_FILES, MAX_CHANGED = 4, 120   # source files touched / lines changed per commit
MAX_EVENTS, EVENT_CONTEXT, MAX_HISTORY_CHARS = 6, 2, 1600
BEFORE_CHARS, AFTER_CHARS = 1500, 700   # file excerpt around the region
NONE_SHARE = 0.25          # share of "no edit here" rows to keep
AI_NOTE = re.compile(r"co-authored-by:[^\n]*(claude|anthropic|copilot|cursor|codex|openai|chatgpt|gemini|google-labs|"
                     r"devin|aider|windsurf|codeium|jules|sweep|tabnine|amazon q|\bbot\b)|generated (with|by) "
                     r"\[?(claude|copilot|cursor|codex|chatgpt|gemini|aider|devin)|\U0001F916", re.I)
BOT = re.compile(r"\[bot\]|dependabot|renovate|github-actions|copilot|claude|cursor|devin|codex|noreply@anthropic", re.I)
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    path: str
    old_start: int  # 1-based (for an insertion: the line it goes after)
    old: list
    new: list


def deepen(root: Path) -> None:
    if git(root, "rev-parse", "--is-shallow-repository").strip() == "true":
        subprocess.run(["git", "-C", str(root), "fetch", "--unshallow", "--quiet", "--no-tags"],
                       capture_output=True, timeout=900)


def commits(root: Path, head: str) -> list[dict]:
    out = git(root, "log", "--no-merges", f"-{MAX_COMMITS}", "--format=%H%x1f%P%x1f%an%x1f%ae%x1f%B%x1e", head)
    rows = []
    for rec in out.split("\x1e"):
        parts = rec.strip("\n").split("\x1f")
        if len(parts) == 5 and parts[1] and " " not in parts[1].strip():
            rows.append(dict(zip(["sha", "parent", "name", "email", "msg"], parts)))
    return rows


def parse_diff(text: str) -> dict[str, list[Hunk]]:
    files, path, cur = {}, None, None
    for line in text.split("\n"):
        if line.startswith("diff --git"):
            path, cur = None, None
        elif line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            if path:
                files.setdefault(path, [])
        elif path and (m := HUNK.match(line)):
            cur = Hunk(path, int(m.group(1)), [], [])
            files[path].append(cur)
        elif cur is not None and line.startswith("-") and not line.startswith("---"):
            cur.old.append(line[1:])
        elif cur is not None and line.startswith("+") and not line.startswith("+++"):
            cur.new.append(line[1:])
    return files


def apply(lines: list[str], h: Hunk, offset: int) -> tuple[list[str], int, int]:
    """lines with hunk h applied (h.old_start is in the ORIGINAL file; offset = lines added so far above it).
    Returns (new lines, index where h's new lines start, new offset)."""
    at = h.old_start + offset if not h.old else h.old_start - 1 + offset
    assert lines[at:at + len(h.old)] == h.old, "hunk doesn't match"
    return lines[:at] + h.new + lines[at + len(h.old):], at, offset + len(h.new) - len(h.old)


def render_event(before: list[str], at: int, h: Hunk) -> str:
    """One past edit as a small diff with EVENT_CONTEXT lines around it (in the file as it was before the edit)."""
    lo, hi = max(0, at - EVENT_CONTEXT), at + len(h.old)
    ctx_after = before[hi:hi + EVENT_CONTEXT]
    body = ([" " + l for l in before[lo:at]] + ["-" + l for l in h.old] + ["+" + l for l in h.new]
            + [" " + l for l in ctx_after])
    return f"--- {h.path}\n@@ line {at + 1} @@\n" + "\n".join(body) + "\n"


def cut_tail(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    text = text[-n:]
    return text[text.find("\n") + 1:]


def cut_head(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[:text.rfind("\n", 0, n) + 1]


def examples_for_commit(rng: random.Random, root: Path, repo: str, c: dict, pats: list) -> list[dict]:
    if BOT.search(c["name"] + " " + c["email"]) or AI_NOTE.search(c["msg"]):
        return []
    names = [p for p in git(root, "diff", "--name-only", "--no-renames", c["parent"], c["sha"]).split("\n") if p]
    src = [p for p in names if is_source(p) and any(pt.fullmatch(p) for pt in pats)]
    if not src or len(src) > MAX_FILES:
        return []
    diff = parse_diff(git(root, "diff", "--no-color", "--no-renames", "-U0", c["parent"], c["sha"], "--", *src))
    changed = sum(len(h.old) + len(h.new) for hs in diff.values() for h in hs)
    if changed > MAX_CHANGED or sum(len(hs) for hs in diff.values()) < 2:
        return []
    rows = []
    for path, hunks in diff.items():
        if len(hunks) < 2:
            continue
        old_text = git(root, "show", f"{c['parent']}:{path}")
        new_text = git(root, "show", f"{c['sha']}:{path}")
        if not old_text or not new_text or GENERATED.search("\n".join(old_text.split("\n")[:10])):
            continue  # file added or deleted by the commit, or generated
        if any(re.sub(r"\s+", "", "".join(h.old)) == re.sub(r"\s+", "", "".join(h.new)) for h in hunks):
            continue  # whitespace-only hunks: formatting, not an edit worth predicting
        states, offset, lines = [], 0, old_text.replace("\r\n", "\n").split("\n")
        try:
            for h in hunks:  # the file before each hunk, and where the hunk lands in it
                before = lines
                lines, at, offset = apply(lines, h, offset)
                states.append((before, at))
        except AssertionError:
            continue
        if "\n".join(lines) != new_text.replace("\r\n", "\n"):
            continue  # the parsed hunks don't rebuild the new file (e.g. "\ No newline at end of file"): skip it
        if any(len(l) > MAX_LINE_CHARS for l in lines):
            continue
        for k in range(1, len(hunks)):
            rows += example(rng, repo, c, path, hunks, states, k, lines)
    return rows


def example(rng, repo, c, path, hunks, states, k, final_lines) -> list[dict]:
    """History = hunks[:k] of this file; the cursor sits at the end of hunk k-1; the region is a window around it.
    "edit" when the next hunks that fall inside the region change it, "none" when no hunk touches it."""
    events = [render_event(b, at, h) for (b, at), h in zip(states[:k], hunks[:k])][-MAX_EVENTS:]
    while len(events) > 1 and sum(map(len, events)) > MAX_HISTORY_CHARS:
        events.pop(0)
    if sum(map(len, events)) > MAX_HISTORY_CHARS:
        return []
    cur_lines, _ = states[k]  # the file with hunks[:k] applied
    _, last_at = states[k - 1]
    cl = last_at + max(len(hunks[k - 1].new), 1) - 1  # cursor line: the last line of the previous edit
    cl = min(cl, len(cur_lines) - 1)
    lo, hi = max(0, cl - rng.randint(3, 10)), min(len(cur_lines), cl + rng.randint(6, 16) + 1)
    # Apply the remaining hunks that sit wholly inside [lo, hi); a hunk on or across the edge makes it ambiguous.
    region_new, inside, offset, shift = list(cur_lines[lo:hi]), 0, 0, 0
    for j in range(k, len(hunks)):
        h, (_, at) = hunks[j], states[j]
        a = at - shift  # states[j] has hunks[k:j] applied too; they sit above hunk j, so undo their line shift
        shift += len(h.new) - len(h.old)
        if h.old:  # replaces or deletes cur_lines[a : a + len(old)]
            if a + len(h.old) <= lo or a >= hi:
                continue
            if a < lo or a + len(h.old) > hi:
                return []
        else:  # inserts before cur_lines[a]
            if a < lo or a > hi:
                continue
            if a == lo or a == hi:
                return []
        r_at = a - lo + offset
        region_new = region_new[:r_at] + h.new + region_new[r_at + len(h.old):]
        offset += len(h.new) - len(h.old)
        inside += 1
    kind = "edit" if inside else "none"
    if kind == "none" and rng.random() > NONE_SHARE / (1 - NONE_SHARE):
        return []
    region = list(cur_lines[lo:hi])
    col = len(region[cl - lo]) if lo <= cl < hi else 0
    region_shown = region[:cl - lo] + [region[cl - lo][:col] + CURSOR + region[cl - lo][col:]] + region[cl - lo + 1:]
    before = cut_tail("\n".join(cur_lines[:lo]) + ("\n" if lo else ""), BEFORE_CHARS)
    after = cut_head(("\n" if hi < len(cur_lines) else "") + "\n".join(cur_lines[hi:]), AFTER_CHARS)
    prompt = (f"<filename>{path}\n{HISTORY}\n" + "".join(events) + f"{EXCERPT}\n{before}{REGION_START}\n"
              + "\n".join(region_shown) + f"\n{REGION_END}{after}\n{REWRITE}\n{REGION_START}\n")
    target = "\n".join(region_new) + f"\n{REGION_END}"
    rid = f"nes_{repo.replace('/', '__')}_{c['sha'][:10]}_{path}_{k}"
    return [{"id": rid, "repo": repo, "commit": c["sha"], "path": path, "lang": EXT_LANG[Path(path).suffix],
             "kind": kind, "prompt": prompt, "target": target}]


def build_repo(job: dict) -> dict:
    root, repo = Path(job["root"]), job["repo"]
    rng = random.Random(f"{job['seed']}:{repo}")
    stats = collections.Counter()
    try:
        deepen(root)
    except subprocess.TimeoutExpired:
        return {"repo": repo, "rows": [], "stats": {"fetch_timeout": 1}}
    pats = [glob_regex(g) for g in job["src"]]
    rows = []
    for c in commits(root, job["commit"]):
        stats["commits"] += 1
        if BOT.search(c["name"] + " " + c["email"]) or AI_NOTE.search(c["msg"]):
            stats["skipped_ai_or_bot"] += 1
            continue
        got = examples_for_commit(rng, root, repo, c, pats)
        stats["commits_used"] += bool(got)
        rows += got
    stats["rows"] = len(rows)
    return {"repo": repo, "rows": rows, "stats": dict(stats)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clones", default=str(Path.home() / "quark_repos"))
    ap.add_argument("--heldout", action="store_true", help="the benchmark repos instead of the training repos")
    ap.add_argument("--repos", type=int, default=0, help="only the first N repos (smoke test)")
    ap.add_argument("--out", default="nes_train.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    clones = Path(a.clones)
    if a.heldout:
        pinned = {json.loads(l)["repo"]: json.loads(l)["commit"] for l in open("quark_repos.jsonl", encoding="utf-8")
                  if l.strip()}
        repos = [{"repo": r, "src": REPOS[r]["src"], "commit": pinned[r]} for r in HELDOUT]
    else:
        repos, _ = load_repos(clones)
        assert not {r["repo"] for r in repos} & set(HELDOUT)
    repos = [r for r in repos if (clones / r["repo"].replace("/", "__")).is_dir()]
    if a.repos:
        repos = repos[:a.repos]
    jobs = [{"repo": r["repo"], "root": str(clones / r["repo"].replace("/", "__")), "src": r["src"],
             "commit": r["commit"], "seed": a.seed} for r in repos]
    total, seen, n = collections.Counter(), set(), 0
    with open(a.out, "w", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.workers) as pool:
        for res in pool.map(build_repo, jobs):
            total.update(res["stats"])
            for row in res["rows"]:
                key = (row["prompt"][-3000:], row["target"])
                if key in seen:
                    continue
                seen.add(key)
                f.write(json.dumps(row) + "\n")
                n += 1
            print(f"{res['repo']}: {res['stats']}", flush=True)
    print(f"\n{n} rows from {len(jobs)} repos -> {a.out}  ({dict(total)})")


if __name__ == "__main__":
    main()
