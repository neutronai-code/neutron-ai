# Task: solve the tasks Hadron's new base can't, and turn your verified answers into training data

You are working in the NeutronAI repo at `C:\Users\manoj\Second Sun` (Windows; the shell is PowerShell). The owner is a
beginner: if they have to do anything by hand, give exact click-by-click steps.

**You are the teacher.** You write the answers yourself, in your own reasoning. Don't call any model API, gateway or
script that asks another model (no `distill.py`, no Onysoft, no OpenRouter, no local server): the point is to capture
*your* step-by-step reasoning on these tasks.

## Background

Hadron is NeutronAI's reasoning and repository-editing model. Its new base is
`JetBrains/Mellum2.1-12B-A2.5B-Thinking` (78.1% on our 236 held-out tasks vs 53.7% for the old base). We measured
which of the 2,871 training tasks Mellum2.1 can already solve (4 tries each). The 304 it **never** solved are in
`distill_targets_Mellum2.1-12B-A2.5B-Thinking-32k.jsonl` (repo root). RL can't learn from a task the model never
gets right, so we fine-tune it first on worked, **test-verified** answers to these tasks: yours.

By language: 87 cpp, 74 python (11 of them repo bug-fixes with `### FILE:` blocks), 60 typescript, 49 javascript,
23 python_stdio, 11 sql.

## The loop

`distill_agent.py` hands you tasks and checks your answers with the same sandbox the benchmark uses. Repeat:

1. `uv run distill_agent.py next --n 5`
   writes the next 5 unsolved tasks to `inbox/distill_next.md`: for each, its id, the system prompt and the user
   prompt, exactly as Hadron will see them. Read that file.
2. Solve each task. Write `inbox/distill_NNN.json` (the file name `next` tells you) as a JSON array:

   ```json
   [{"id": "gen_cpp_...", "think": "<your step-by-step reasoning>", "answer": "```cpp\n...\n```"}]
   ```

   - `answer`: exactly the format the task's system prompt asks for: one fenced code block in the task's language,
     or for repo tasks `### FILE: <path>` followed by `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` blocks with
     the SEARCH text copied exactly from the file. Nothing else in the answer.
   - `think`: the reasoning that gets from the prompt to the answer, written the way you'd want a smaller model to
     learn to think. Restate what's asked, pin down the tricky rules and edge cases from the prompt, then **derive**
     the answer: set out the approach, work a concrete example or two by hand, walk the edge cases, and say why the
     code is correct. Repeating the prompt back is worth nothing, and the ingest check measures exactly that: if
     your reasoning mostly restates the prompt, it's rejected. No `<think>` tags (they're added for you).
     600 to 40,000 characters; most tasks need 1,500-8,000.
   - Write valid JSON: escape newlines as `\n` and quotes as `\"` inside strings. Writing the file with a small
     Python script (`json.dump`) is the easiest way to get this right.
3. `uv run distill_agent.py ingest inbox/distill_NNN.json --teacher <your model name>`
   (for example `--teacher deepseek-v4.1-flash`; use your real model name). Each answer is run against the task's
   tests: `PASS` rows are kept in `sft_distill_mellum.jsonl`, `FAIL` rows print why.
4. A failed task stays in the queue, and the next `next` shows it again with the error and your last answer. Fix it
   in a later batch; reason it out again rather than just patching the symptom, and write the `think` as a clean
   solve (not "my last attempt failed because...": Hadron never sees that attempt).
5. `uv run distill_agent.py stats` shows progress by language.

At the end (or any time you want to check the data), run:

- `uv run distill_agent.py recheck`
  re-verifies every kept answer in the real sandbox and re-checks each trace's quality. `BAD` rows must go; `weak`
  rows (thin reasoning, or reasoning that only restates the prompt) are usable but could be better.
- `uv run distill_agent.py recheck --drop-bad --drop-weak`
  removes them, so those tasks go back in the queue for another attempt.

Keep going until every task is solved or skipped. Do batches of 3-5 tasks; `--lang cpp` (etc.) on `next` works one
language at a time if that's easier.

## Rules for the answers (important)

- **Solve from the prompt only.** Don't open the targets file, and never look at a task's `test`,
  `oracle_solution` or `oracle_patch` fields, or at any other copy of the task in the repo (`train_clean.jsonl`,
  `tasks_generated.jsonl`, ...). Hadron gets only the prompt, so the reasoning must work from the prompt alone.
  `ingest` rejects answers identical to the stored reference, `think` text that mentions an oracle, reference
  solution or hidden tests, and `think` text that talks about the test, the harness or the task id.
- When a failure message reveals what a test expects, use it to find your mistake in reading the prompt, then
  write the reasoning as if you'd read it right the first time. Don't special-case test inputs, and don't mention
  the test in the reasoning.
- `uv run distill_agent.py skip <id> --reason "..."` only when a task truly can't be solved as written (the prompt
  contradicts itself, or the failure shows the tests demand something the prompt rules out). Quote the exact prompt
  sentence and test expectation that cannot both hold. An under-specified constant, return schema or event schema
  is also a valid skip, but say exactly what is missing.
- **Never skip for a vague reason** such as "ambiguous edge case", "unstated rule" or "undocumented structure".
  Natural validation (reversed intervals, duplicate ids, non-string values) is not a contradiction: implement it and
  solve the task. A failed attempt or an unusual test expectation is not evidence by itself; identify the concrete
  behavior that makes the task impossible or ambiguous. Not for tasks that are just hard: those are the ones we need most.
- **Do not re-skip a task that was requeued after review.** Its latest attempt note gives the exact missing behavior;
  implement it. If you still think the task is broken, stop and tell the owner instead of skipping it again.
- Don't edit `distill_agent.py`, `sandbox.py`, `repair.py`, the targets file, `tasks_holdout.jsonl`, `tasks.jsonl`,
  `hadron_score.py` or the molab notebooks. If the checker seems wrong, stop and tell the owner what you saw.

## Repo rules

- Never print, log or commit API keys or tokens. Before any commit, scan the staged changes for real credentials,
  including the owner's known key fragments from your local instructions and any Hugging Face token, and stop if
  anything matches.
- Commit or push only when the owner says so. Keep data files (`*.jsonl`, `inbox/`) out of git.
- Never use the owner's private TankFlare code for anything.
- Don't upload the results to Hugging Face: `neutronai-code/hadron-data` is a **public** dataset. Ask the owner first.

## When you stop, report back (short)

- `uv run distill_agent.py stats` output (solved / skipped / to do, by language; think lengths).
- Every skipped task with its reason.
- Leak check: `uv run audit.py --train sft_distill_mellum.jsonl --eval tasks_holdout.jsonl --clean-out sft_distill_mellum_clean.jsonl`
  and say how many rows it flagged.
- One example row's id.
