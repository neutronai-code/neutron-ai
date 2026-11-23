"""The repo-context versions of the real-repo benchmark: the same 600 holes as repo_holes.jsonl (same windows, same
tests), but each prompt first shows other code of the same repo as `<|file_sep|>path` blocks:

  --gatherer imports (v0.4's training format, quark_data_v04.context_block): API skeletons of up to 4 related files
      (imports, then importers, else the same folder)                                   -> repo_holes_ctx.jsonl
  --gatherer names (ctx_gather.py): signatures picked by name, the way an editor plugin would: definitions of the names
      around the cursor, members of the enclosing class, what the imports provide, an outline of the current file
      outside the window                                                                 -> repo_holes_ctx2.jsonl

The IDE's lighter prompt profiles for slow machines: the same holes and tests with smaller budgets. --prefix-chars and
--suffix-chars re-cut the hole's window at line boundaries, as repo_fim.window does with 2,400 / 900:
  --budget 2000 --prefix-chars 1600 --suffix-chars 600 --out repo_holes_light.jsonl      (Quark-12B "Light")
  --budget 2000 --prefix-chars 1600 --suffix-chars 301 --out repo_holes_fastlight.jsonl  (Quark-1.5B "Fast" + Light:
      its prompt is prefix-first, so the IDE keeps the suffix to the rest of the line plus <=300 chars of whole lines)

Neither ever sees the missing code. The other files come from the local clones (~/quark_repos, checked to be at the
pinned commits). Same row ids, so a hole can be compared across prompt styles.

  uv run python make_ctx_bench.py --gatherer names
  modal run modal_repo_fim.py --eval --data repo_holes_ctx2.jsonl --report repo_eval_report_ctx2.jsonl --models ...
"""
import argparse
import collections
import json
import random
from pathlib import Path

from ctx_gather import RepoContext
from quark_data_v04 import context_block, git, import_graph, repo_files
from fim import format_qwen_fim_prompt
from repo_fim import PREFIX_CHARS, REPOS, SUFFIX_CHARS


def window(lines: list[str], lo: int, hi: int, prefix_chars: int, suffix_chars: int) -> tuple[str, str]:
    """repo_fim.window with other sizes: up to prefix_chars before the hole and suffix_chars after it, line-cut."""
    prefix = "".join(lines[:lo])
    if len(prefix) > prefix_chars:
        prefix = prefix[-prefix_chars:]
        prefix = prefix[prefix.find("\n") + 1:]
    suffix = "\n" + "".join(lines[hi:])
    if len(suffix) > suffix_chars:
        suffix = suffix[:suffix.rfind("\n", 0, suffix_chars) + 1]
    return prefix, suffix


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clones", default=str(Path.home() / "quark_repos"))
    ap.add_argument("--holes", default="repo_holes.jsonl")
    ap.add_argument("--gatherer", choices=["imports", "names"], default="names")
    ap.add_argument("--budget", type=int, default=4000, help="context characters (names gatherer)")
    ap.add_argument("--prefix-chars", type=int, default=PREFIX_CHARS, help="names gatherer: prompt chars before the hole")
    ap.add_argument("--suffix-chars", type=int, default=SUFFIX_CHARS, help="names gatherer: prompt chars after the hole")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or {"imports": "repo_holes_ctx.jsonl", "names": "repo_holes_ctx2.jsonl"}[a.gatherer]
    rows = [json.loads(line) for line in open(a.holes, encoding="utf-8") if line.strip()]
    repos = {}
    for repo in sorted({r["repo"] for r in rows}):
        root = Path(a.clones) / repo.replace("/", "__")
        commit = next(r["commit"] for r in rows if r["repo"] == repo)
        head = git(root, "rev-parse", "HEAD").strip()
        if head != commit:
            raise SystemExit(f"{repo}: clone is at {head[:10] or 'nothing'}, the holes are from {commit[:10]}")
        files = repo_files(root, REPOS[repo]["src"])
        for r in rows:  # a hole's own file must be in the graph even if repo_files filtered it (its imports count)
            if r["repo"] == repo and r["path"] not in files:
                files[r["path"]] = (root / r["path"]).read_text(encoding="utf-8").replace("\r\n", "\n")
        if a.gatherer == "names":
            repos[repo] = RepoContext(files)
        else:
            graph = import_graph(files)
            callers = collections.defaultdict(set)
            for f, deps in graph.items():
                for d in deps:
                    callers[d].add(f)
            repos[repo] = files, graph, callers

    n_ctx, sizes = 0, []
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            if a.gatherer == "names":
                rc = repos[r["repo"]]
                lines = rc.files[r["path"]].splitlines(keepends=True)
                before, after = "".join(lines[:r["lo"]]), "\n" + "".join(lines[r["hi"]:])
                editor = before + after  # the file with the hole's lines missing: what the editor has
                cur = len(before)
                assert editor[cur - len(r["prefix"]):cur] == r["prefix"].replace("\r\n", "\n"), r["id"]
                if (a.prefix_chars, a.suffix_chars) != (PREFIX_CHARS, SUFFIX_CHARS):  # a smaller window: re-cut
                    prefix, suffix = window(lines, r["lo"], r["hi"], a.prefix_chars, a.suffix_chars)
                    assert r["prefix"].replace("\r\n", "\n").endswith(prefix), r["id"]
                    r = {**r, "prefix": prefix, "suffix": suffix, "fim_prompt": format_qwen_fim_prompt(prefix, suffix)}
                ctx = rc.gather(r["path"], editor, cur - len(r["prefix"]), cur + len(r["suffix"]), cur, a.budget)
            else:
                files, graph, callers = repos[r["repo"]]
                ctx = context_block(random.Random(r["id"]), r["path"], files, graph, callers)
            n_ctx += bool(ctx)
            sizes.append(len(ctx))
            f.write(json.dumps({**r, "ctx_chars": len(ctx), "fim_prompt": ctx + r["fim_prompt"]}) + "\n")
    by_repo = collections.Counter(r["repo"] for r, s in zip(rows, sizes) if s)
    print(f"{n_ctx}/{len(rows)} holes got repo context (avg {sum(sizes) / max(1, n_ctx):.0f} chars) -> {out}")
    for repo in sorted({r["repo"] for r in rows}):
        print(f"  {repo:28} {by_repo[repo]:3}/{sum(r['repo'] == repo for r in rows)}")


if __name__ == "__main__":
    main()
