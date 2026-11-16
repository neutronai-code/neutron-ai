# Brief: generate verified training tasks for NeutronAI

You are helping build a dataset of programming tasks. Every task you write is checked automatically by
`cases.py`, which runs your reference solution against your tests in a sandbox. Only tasks that pass all
checks are kept. Your goal is to add as many **correct, varied, hard** tasks as possible.

## The loop

1. See what exists and where the gaps are:
   `uv run cases.py stats`
2. Pick the language furthest below its Round 2 target (see "Round 2" below). Get the exact format spec for a batch:
   `uv run cases.py prompt --lang <lang> --n 10 --out-file inbox/spec_<lang>.md`
   Read `inbox/spec_<lang>.md` and follow it exactly. It lists the required JSON keys and per-language test rules.
3. Write the batch as a JSON array to a **new** file: `inbox/hard_<lang>_<batch number>.json`
   (for example `inbox/hard_cpp_007.json`). Never overwrite or edit an earlier batch file.
4. Check it:
   `uv run cases.py ingest inbox/hard_<lang>_<batch number>.json --source <your model name>-hard --min-lines 12`
5. Read why cases were dropped (printed, and in `cases_rejected.jsonl`). Fix the **solution** (or replace the
   case) and put the corrected cases in the **next** batch file, then ingest that file.
6. Repeat until every Round 2 target is met. Run `uv run cases.py stats` every few batches.

## Round 2: HARD cases only

Round 1 is done (1807 cases, every target met). Many of those turned out too easy: a small model solves them every
time, so it learns nothing from them. **Round 2 is about difficulty, not count.** A case is useful only if a strong
4B-parameter coding model, thinking first, would *sometimes get it wrong*.

Name round-2 batch files `inbox/hard_<lang>_<NNN>.json` (start at 001) and always ingest with
`--source <your model name>-hard --min-lines 12`. `--min-lines 12` drops any case whose reference solution has fewer
than 12 lines of code (not applied to sql or repo_python). `uv run cases.py stats` shows round-2 counts under
"By source".

| lang | new hard cases |
|---|---|
| python | 250 |
| python_stdio | 200 |
| cpp | 250 |
| repo_python | 200 |
| typescript | 150 |
| javascript | 150 |
| sql | 100 |

### What makes a case hard enough

- **Several steps, not one trick.** Parse input, keep state, then run an algorithm; or two interacting rules. If you
  could write the solution in a minute without thinking, skip it.
- **Good areas:** graphs (shortest paths with twists, topological order with tie rules, union-find), dynamic
  programming, intervals and sweep lines, heaps and scheduling simulations, binary search on the answer, tries and
  string algorithms, tokenizers/parsers/expression evaluators, caches with eviction rules (LRU/LFU/TTL), rate limiters,
  calendar/date arithmetic without libraries, money rounding and splitting, diff/merge, glob or semver-range matching,
  CSV with quoting and escapes, state machines.
- **Edge cases a quick first attempt misses:** empty input, duplicates, ties and their ordering rule, negative numbers,
  64-bit overflow (C++), off-by-one boundaries, unusual but valid characters. State every rule in the prompt; the tests
  must only check what the prompt says.
- **Tests that catch near-misses:** at least 8 checks, including one or more inputs big enough that a naive O(n^2)
  solution is clearly slow (python_stdio: n around 10^5 in one test, still under 100 KB).
- **repo_python:** bugs that need reading 2+ files to find (wrong cache invalidation, pagination off-by-one, a helper
  used with the wrong units, an ordering bug in a pipeline), with tests that fail before the fix.
- **sql:** window functions, recursive CTEs, gaps-and-islands, ties in rankings, NULL handling in joins/aggregates.
- **Avoid:** one-line string helpers, wrappers around built-ins (sum/sort/split/upper), simple counting or
  filtering, renamed copies of earlier cases.

## Round 1 targets (done)

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
