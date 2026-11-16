# Brief: generate verified training tasks for NeutronAI

You are helping build a dataset of programming tasks. Every task you write is checked automatically by
`cases.py`, which runs your reference solution against your tests in a sandbox. Only tasks that pass all
checks are kept. Your goal is to add as many **correct, varied, hard** tasks as possible.

## The loop

1. See what exists and where the gaps are:
   `uv run cases.py stats`
2. Pick the language furthest below its target (see Targets). Get the exact format spec for a batch:
   `uv run cases.py prompt --lang <lang> --n 20 --out-file inbox/spec_<lang>.md`
   Read `inbox/spec_<lang>.md` and follow it exactly. It lists the required JSON keys and per-language test rules.
3. Write the batch as a JSON array to a **new** file: `inbox/<lang>_<batch number>.json`
   (for example `inbox/cpp_007.json`). Never overwrite or edit an earlier batch file.
4. Check it:
   `uv run cases.py ingest inbox/<lang>_<batch number>.json --source <your model name>`
5. Read why cases were dropped (printed, and in `cases_rejected.jsonl`). Fix the **solution** (or replace the
   case) and put the corrected cases in the **next** batch file, then ingest that file.
6. Repeat until every target is met. Run `uv run cases.py stats` every few batches.

## Targets

| lang | target |
|---|---|
| repo_python | 300 |
| cpp | 300 |
| typescript | 300 |
| javascript | 300 |
| sql | 200 |
| python | 300 |
| python_stdio | 100 |

## Rules (important)

- Only create files inside `inbox/`. Do **not** edit `cases.py`, `sandbox.py`, `audit.py`, `fuzz_tests.py`,
  or any other existing file, and never write to `tasks_generated.jsonl` or `cases_rejected.jsonl` directly.
  Every kept task must come through `cases.py ingest`. (They are all re-verified before training anyway.)
- Never weaken or delete tests to make a solution pass. If the reference solution fails, fix the solution or
  drop the case.
- Do not copy or lightly reword problems from public benchmarks (LiveCodeBench, HumanEval, MBPP, APPS,
  Codeforces, LeetCode). Invent new problems: they are used to *evaluate* models and must stay unseen.
- Make every case different: new idea, not the same problem with renamed variables. Prefer realistic,
  tricky tasks with real edge cases over toy exercises.
- Do not run git commands (no commit, push, or checkout).
