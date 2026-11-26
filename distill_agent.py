"""Agent-written distillation: a coding agent (the teacher model itself) solves the tasks Hadron's base can't,
and only answers that pass the task's tests are kept as SFT traces. The cases.py loop, for solutions.

No API calls: the agent reads a few tasks, writes a batch file with its own reasoning + answer for each, and ingests
it. Every answer runs through sandbox.verify (the same check as eval.py); failures come back with the error and are
shown again on the task the next time `next` lists it, so the agent can fix them in a later batch.

  uv run distill_agent.py next --n 5                                   # next unsolved tasks -> inbox/distill_next.md
  uv run distill_agent.py ingest inbox/distill_001.json --teacher <model>  # check a batch, keep the passing ones
  uv run distill_agent.py stats
  uv run distill_agent.py recheck --drop-bad                           # re-verify kept rows; drop the weak ones
  uv run distill_agent.py skip <task id> --reason "the prompt is ambiguous about ties"

Batch file: a JSON array of {"id": ..., "think": "<step-by-step reasoning>", "answer": "<final answer in the format
the task's system prompt asks for>"}. Kept rows go to --out in train.py's SFT format:
{"id", "teacher", "messages": [system, user, assistant]}, the assistant turn being
"<think>\\n{think}\\n</think>\\n\\n{answer}" and the user turn repair.task_prompt(task), exactly what Hadron sees.

`next` never shows a task's tests or reference solution: Hadron won't see them either, so a trace that leans on them
would teach it nothing it can use.
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repair import system_prompt_for, task_prompt
from sandbox import verify

DEFAULT_TARGETS = "distill_targets_Mellum2.1-12B-A2.5B-Thinking-32k.jsonl"
DEFAULT_OUT = "sft_distill_mellum.jsonl"
MAX_THINK_CHARS = 40000  # ~11K tokens: every trace must fit Hadron's training window with its prompt and answer
MIN_THINK_CHARS = 600    # under this the "reasoning" is a requirements list, not reasoning worth training on
MIN_NOVEL_CHARS = 200    # under this the reasoning adds nothing the prompt didn't already say
LEAK_WORDS = re.compile(r"\b(oracle|reference solution|the hidden tests?|test file|distill_targets)\b", re.IGNORECASE)
# A trace must reason about the PROBLEM. Talking about the test, the task id, or the harness is the signature of
# an answer that was fitted to the tests instead of derived from the prompt, so it teaches Hadron nothing.
TEST_TALK = re.compile(
    r"(?:\bthe tests?\b|\btest\s+\d\b|\bassert(?:ion)?s?\b|\bharness\b|\bexpected value|"
    r"\bAssertionError\b|\bgen_[a-z]+_[0-9a-f]{6,}\b|\bfailing (?:assert|test)\b|\bthe test (?:says|expects|requires)\b)",
    re.IGNORECASE,
)
# A trace that only lists the requirements teaches "restate the prompt, then emit code". Measuring that with a
# keyword regex is unreliable, so measure it directly: how much of the reasoning is NEW content, rather than
# lines the prompt already said? A trace that adds nothing teaches nothing.
_WORD = re.compile(r"[^a-z0-9]+")


def novel_chars(think: str, prompt: str) -> int:
    """Characters of the reasoning that aren't a restatement of the prompt."""
    p = _WORD.sub(" ", prompt.lower())
    p = " " + " ".join(p.split()) + " "
    total = 0
    for line in think.splitlines():
        n = " ".join(_WORD.sub(" ", line.lower()).split())
        if len(n) < 12:            # too short to compare
            total += len(line)
            continue
        if " " + n + " " in p:      # the whole line is in the prompt
            continue
        if all(w in p for w in n.split()):  # only uses the prompt's words
            continue
        total += len(line)
    return total


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def paths(args):
    out = os.path.join(HERE, args.out)
    return out, out + ".skipped.jsonl", out + ".attempts.jsonl"


def load(args):
    targets = read_jsonl(os.path.join(HERE, args.targets))
    if not targets:
        sys.exit(f"{args.targets} not found or empty: it is made by `hadron_score.py <TAG> --set train`.")
    held = {t["id"] for name in ("tasks_holdout.jsonl", "tasks.jsonl") for t in read_jsonl(os.path.join(HERE, name))}
    leaked = [t["id"] for t in targets if t["id"] in held]
    if leaked:
        sys.exit(f"{len(leaked)} target tasks are benchmark tasks ({leaked[:3]}): refusing to distill them.")
    out, skipped_path, attempts_path = paths(args)
    done = {r["id"] for r in read_jsonl(out)}
    skipped = {r["id"] for r in read_jsonl(skipped_path)}
    attempts = collections.defaultdict(list)
    for a in read_jsonl(attempts_path):
        attempts[a["id"]].append(a)
    return targets, done, skipped, attempts


def files_of(task):
    files = task.get("files")
    if isinstance(files, str):
        files = json.loads(files) if files else None
    return files or None


def answer_format(task):
    if files_of(task):
        return "`### FILE: <path>` + `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` blocks (SEARCH copied exactly)"
    fence = {"python_stdio": "python", "sql": "sql", "cpp": "cpp", "typescript": "typescript",
             "javascript": "javascript"}.get((task.get("lang") or "python").lower(), "python")
    return f"one ```{fence} fenced code block"


def cmd_next(args):
    targets, done, skipped, attempts = load(args)
    todo = [t for t in targets if t["id"] not in done and t["id"] not in skipped]
    if args.lang:
        todo = [t for t in todo if (t.get("lang") or "") == args.lang]
    todo = todo[: args.n]
    if not todo:
        print("Nothing left: every target task is solved or skipped. Run `stats`.")
        return
    inbox = os.path.join(HERE, "inbox")
    os.makedirs(inbox, exist_ok=True)
    n_batch = len([f for f in os.listdir(inbox) if re.fullmatch(r"distill_\d+\.json", f)]) + 1
    lines = [f"# {len(todo)} tasks to solve", "",
             f"Write your answers to `inbox/distill_{n_batch:03d}.json` as a JSON array of",
             '`{"id": ..., "think": ..., "answer": ...}`, then run',
             f"`uv run distill_agent.py ingest inbox/distill_{n_batch:03d}.json --teacher <your model name>`.", ""]
    for t in todo:
        lines += ["---", "", f"## `{t['id']}`  ({t.get('lang', 'python')})", "",
                  f"**Answer format:** {answer_format(t)}", "", "**System prompt:**", "", "```text",
                  system_prompt_for(t), "```", "", "**User prompt:**", "", "````text", task_prompt(t), "````", ""]
        prev = attempts.get(t["id"])
        if prev:
            lines += [f"**Your earlier attempts failed ({len(prev)}x).** Latest result: `{prev[-1]['why'][:300]}`", ""]
            lines += ["Previous answer:", "", "````text", prev[-1].get("answer", "")[:4000], "````", ""]
    path = os.path.join(inbox, "distill_next.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    left = len([t for t in targets if t["id"] not in done and t["id"] not in skipped])
    print(f"Wrote {len(todo)} tasks to {os.path.relpath(path, HERE)}  ({left} unsolved in total). "
          f"Answer them in inbox/distill_{n_batch:03d}.json.")


def parse_batch(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n(.*)\n```", text, re.DOTALL)
    if m and not text.startswith("["):
        text = m.group(1)
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise TypeError("the batch must be a JSON array of {id, think, answer}")
    return data


def norm_code(s):
    return re.sub(r"\s+", "", re.sub(r"```\w*", "", s or ""))


def check_item(item, task):
    """(ok, why) before running anything: shape, length and format checks."""
    think, answer = item.get("think"), item.get("answer")
    if not isinstance(think, str) or not isinstance(answer, str) or not answer.strip():
        return False, "needs string fields think and answer"
    think, answer = think.strip(), answer.strip()
    if len(think) < MIN_THINK_CHARS:
        return False, f"think is too short ({len(think)} chars < {MIN_THINK_CHARS}): show the real reasoning"
    if len(think) > MAX_THINK_CHARS:
        return False, f"think is too long ({len(think)} chars > {MAX_THINK_CHARS})"
    if "<think>" in think + answer or "</think>" in think + answer:
        return False, "don't write <think> tags yourself; they are added for you"
    if LEAK_WORDS.search(think):
        return False, f"think mentions {LEAK_WORDS.search(think).group(0)!r}: reason only from the prompt Hadron sees"
    if files_of(task):
        if "### FILE:" not in answer or "<<<<<<< SEARCH" not in answer:
            return False, "repo tasks need ### FILE: + SEARCH/REPLACE blocks"
    elif "```" not in answer:
        return False, "the answer must be a fenced code block"
    ref = task.get("oracle_solution") or task.get("oracle_patch") or ""
    if ref and norm_code(ref) == norm_code(answer):
        return False, "the answer is the stored reference solution: write your own"
    fatal, weak = trace_problems(think, task["prompt"])
    if fatal:
        return False, fatal
    if weak:
        return False, f"{weak} (write it properly, then answer again)"
    return True, "ok"


def trace_problems(think: str, prompt: str = "") -> tuple[str, str]:
    """(fatal, weak) problems with a reasoning trace. Fatal = never train on it. Weak = usable but thin."""
    fatal = ""
    m = TEST_TALK.search(think)
    if m:
        fatal = f"the reasoning talks about the test/harness ('{m.group(0)}'): reason from the prompt only"
    weak = ""
    if len(think) < MIN_THINK_CHARS:
        weak = f"the reasoning is thin ({len(think)} chars < {MIN_THINK_CHARS})"
    elif prompt and novel_chars(think, prompt) < MIN_NOVEL_CHARS:
        weak = (f"the reasoning mostly restates the prompt ({novel_chars(think, prompt)} chars of new content "
                f"< {MIN_NOVEL_CHARS}): derive the answer instead of repeating the prompt")
    return fatal, weak


def cmd_ingest(args):
    targets, done, skipped, _ = load(args)
    by_id = {t["id"]: t for t in targets}
    out, _, attempts_path = paths(args)
    with open(args.batch, encoding="utf-8") as f:
        try:
            items = parse_batch(f.read())
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            sys.exit(f"Couldn't read {args.batch}: {e}. It must be a JSON array of {{id, think, answer}}.")
    kept, failed = [], []
    for item in items:
        tid = item.get("id") if isinstance(item, dict) else None
        task = by_id.get(tid)
        if task is None:
            print(f"  ????  {str(tid)[:40]:40s} not a target task id")
            continue
        if tid in done:
            print(f"  ----  {tid:40s} already solved, ignored")
            continue
        ok, why = check_item(item, task)
        if ok:
            score, info = verify(item["answer"].strip(), task["test"], task.get("starter_code") or "", files_of(task),
                                 task.get("lang", "python"), timeout=args.timeout, use_cache=False,
                                 perf_budget_ms=float(task.get("perf_budget_ms", 0.0) or 0.0))
            if score != 1.0:
                ok = False
                why = f"failed: score {score}, {info.get('passed', '?')}/{info.get('total', '?')} checks passed, " \
                      f"status: {str(info.get('status'))[:240]}"
        if ok:
            think, answer = item["think"].strip(), item["answer"].strip()
            kept.append({"id": tid, "teacher": args.teacher, "messages": [
                {"role": "system", "content": system_prompt_for(task)},
                {"role": "user", "content": task_prompt(task)},
                {"role": "assistant", "content": f"<think>\n{think}\n</think>\n\n{answer}"}]})
            done.add(tid)
            print(f"  PASS  {tid:40s} ({task.get('lang')})")
        else:
            copied = why.startswith("the answer is the stored reference")  # never echo the reference back via `next`
            failed.append({"id": tid, "teacher": args.teacher, "why": why,
                           "answer": "" if copied else str(item.get("answer", ""))})
            print(f"  FAIL  {tid:40s} {why[:160]}")
    append_jsonl(out, kept)
    append_jsonl(attempts_path, failed)
    left = len([t for t in targets if t["id"] not in done and t["id"] not in skipped])
    print(f"\n{len(kept)} kept, {len(failed)} failed -> {args.out} has {len(done)} rows; {left} target tasks left.")
    if failed:
        print("Failed tasks stay in the queue; `next` shows them again with the error and your last answer.")


def cmd_stats(args):
    targets, done, skipped, attempts = load(args)
    out, _, _ = paths(args)
    rows = read_jsonl(out)
    by_lang = collections.defaultdict(collections.Counter)
    for t in targets:
        st = "solved" if t["id"] in done else "skipped" if t["id"] in skipped else "todo"
        by_lang[t.get("lang", "?")][st] += 1
    print(f"{len(targets)} target tasks: {len(done)} solved, {len(skipped)} skipped, "
          f"{len(targets) - len(done) - len(skipped)} to do\n")
    print(f"  {'lang':14s} {'solved':>7s} {'skipped':>8s} {'todo':>6s}")
    for lang, c in sorted(by_lang.items()):
        print(f"  {lang:14s} {c['solved']:7d} {c['skipped']:8d} {c['todo']:6d}")
    if rows:
        lens = [len(r["messages"][-1]["content"].split("</think>")[0]) for r in rows]
        first_try = sum(1 for r in rows if r["id"] not in attempts)
        print(f"\nthink length (chars): median {int(statistics.median(lens))}, max {max(lens)}")
        print(f"solved on the first try: {first_try}/{len(rows)}; failed attempts logged: "
              f"{sum(len(v) for v in attempts.values())}")
        print(f"teachers: {dict(collections.Counter(r.get('teacher') for r in rows))}")


def cmd_skip(args):
    targets, done, skipped, _ = load(args)
    if args.id not in {t["id"] for t in targets}:
        sys.exit(f"{args.id} is not a target task id.")
    if args.id in done or args.id in skipped:
        sys.exit(f"{args.id} is already {'solved' if args.id in done else 'skipped'}.")
    append_jsonl(paths(args)[1], [{"id": args.id, "reason": args.reason}])
    print(f"Skipped {args.id}: {args.reason}")


def cmd_recheck(args):
    """Re-verify every kept row from scratch and re-check the reasoning quality of the trace."""
    targets, _, _, _ = load(args)
    by_id = {t["id"]: t for t in targets}
    out, _, _ = paths(args)
    rows = read_jsonl(out)
    good, dropped, weak_rows = [], [], []
    for r in rows:
        t = by_id.get(r["id"])
        if t is None:
            dropped.append((r["id"], "not a target task id"))
            continue
        m = r["messages"]
        if m[0]["content"] != system_prompt_for(t) or m[1]["content"] != task_prompt(t):
            dropped.append((r["id"], "the prompt doesn't match what Hadron sees"))
            continue
        think, answer = split_trace(m[2]["content"])
        score, info = verify(answer, t["test"], t.get("starter_code") or "", files_of(t), t.get("lang", "python"),
                             timeout=args.timeout, use_cache=False,
                             perf_budget_ms=float(t.get("perf_budget_ms", 0.0) or 0.0))
        if score != 1.0:
            dropped.append((r["id"], f"no longer passes: {score} ({str(info.get('status'))[:80]})"))
            continue
        fatal, weak = trace_problems(think, t["prompt"])
        if fatal:
            dropped.append((r["id"], fatal))
            continue
        if weak:
            weak_rows.append((r["id"], weak))
        good.append(r)
    for tid, why in dropped:
        print(f"  {'drop' if args.drop_bad else 'BAD '} {tid:40s} {why[:110]}")
    for tid, why in weak_rows:
        print(f"  weak  {tid:40s} {why[:110]}")
    removed = list(dropped)
    if args.drop_bad and args.drop_weak:
        weak_ids = {tid for tid, _ in weak_rows}
        good = [r for r in good if r["id"] not in weak_ids]
        removed += weak_rows
    if args.drop_bad and removed:
        with open(out, "w", encoding="utf-8") as f:
            for r in good:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # the removed tasks go back in the queue: log the attempt so `next` shows the reason
        append_jsonl(paths(args)[2], [{"id": tid, "teacher": "", "why": f"recheck: {why}", "answer": ""}
                                      for tid, why in removed])
    print(f"\nRechecked {len(rows)} rows: {len(good)} still good, {len(dropped)} bad, {len(weak_rows)} weak"
          + (f" ({len(removed)} removed from {args.out})" if args.drop_bad and removed else ""))


def split_trace(content: str) -> tuple[str, str]:
    """(think, answer) from a saved assistant turn."""
    if "\n</think>\n\n" in content:
        think, answer = content.split("\n</think>\n\n", 1)
        return think.replace("<think>\n", "", 1).strip(), answer.strip()
    return "", content.strip()


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--targets", default=DEFAULT_TARGETS)
    common.add_argument("--out", default=DEFAULT_OUT)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("next", parents=[common], help="write the next unsolved tasks to inbox/distill_next.md")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--lang", default="", help="only this lang (python, cpp, typescript, javascript, sql, python_stdio)")
    i = sub.add_parser("ingest", parents=[common], help="verify a batch file and keep the passing answers")
    i.add_argument("batch")
    i.add_argument("--teacher", required=True, help="the model that wrote the answers, e.g. deepseek-v4.1-flash")
    i.add_argument("--timeout", type=float, default=8.0)
    sub.add_parser("stats", parents=[common], help="progress by language, think lengths, teachers")
    r = sub.add_parser("recheck", parents=[common], help="re-verify every kept row and re-check the trace quality")
    r.add_argument("--drop-bad", action="store_true", help="remove the flagged rows so the tasks go back in the queue")
    r.add_argument("--drop-weak", action="store_true",
                   help="also remove the rows whose reasoning is thin (they stay usable, but you can redo them)")
    r.add_argument("--timeout", type=float, default=8.0)
    s = sub.add_parser("skip", parents=[common], help="give up on a task that can't be solved as written")
    s.add_argument("id")
    s.add_argument("--reason", required=True)
    args = ap.parse_args()
    {"next": cmd_next, "ingest": cmd_ingest, "stats": cmd_stats, "skip": cmd_skip, "recheck": cmd_recheck}[args.cmd](args)


if __name__ == "__main__":
    main()
