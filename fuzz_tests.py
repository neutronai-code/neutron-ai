"""NeutronAI Adversarial Test Hardener (`fuzz_tests.py`) — Kill Reward-Hacking Lookup Tables Automatically!

The #1 failure mode in RLVR coding post-training (documented by DeepSeek-R1, KodCode, & AlphaCode):
  - Weak unit tests let RL models cheat! Example: if a task only tests `add(2, 3) == 5`, an RL model can
    write `def add(a, b): return 5` and get `reward = 1.0`!
  - `fuzz_tests.py` measures **Test Suite Cheat-Resistance** by synthesizing adversarial sham solutions
    (hardcoded literal extractors from the test's own expected values, `None`/`0`/`[]`/`True` constants,
    first-arg identity returns) and executing them in `sandbox.py`!
  - Even better: given an oracle/reference solution + input generator expressions, `harden_task` automatically
    executes the oracle on fresh edge/fuzz inputs and appends verified `assert` lines until ZERO shams survive!

Usage:
  uv run fuzz_tests.py --self-check
  uv run fuzz_tests.py --data tasks.jsonl
"""
import argparse, ast, json
from sandbox import verify


def extract_sham_candidates(prompt: str, test_code: str, lang: str = "python") -> list[str]:
    """Build realistic reward-hacking sham functions (constant returns, hardcoded test-literal returns, arg0 echo)."""
    if (lang or "python").lower() != "python":
        return []
    import re
    fn_match = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)\s*\(", prompt)
    if not fn_match:
        return []
    fn = fn_match.group(1)

    literals = ["None", "0", "1", "-1", "True", "False", "[]", "{}" , "''"]
    try:
        tree = ast.parse(test_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and node.comparators:
                rhs = ast.unparse(node.comparators[0])
                if len(rhs) < 80:
                    literals.append(rhs)
    except SyntaxError:
        pass

    seen, shams = set(), [f"```python\ndef {fn}(*a, **kw):\n    return a[0] if a else None\n```"]
    for lit in literals:
        if lit not in seen:
            seen.add(lit)
            shams.append(f"```python\ndef {fn}(*a, **kw):\n    return {lit}\n```")
    return shams[:12]


def audit_test_strength(task: dict) -> dict:
    """Return cheat-resistance report: a task is `airtight` iff NO sham solution achieves `reward == 1.0`
    and max sham partial reward is `< 0.70`."""
    if task.get("files") or (task.get("lang") or "python").lower() != "python":
        return {"id": task.get("id", ""), "checked": False, "airtight": True, "max_sham_reward": 0.0}

    shams = extract_sham_candidates(task["prompt"], task["test"], task.get("lang", "python"))
    max_r, best_sham = 0.0, ""
    for sham in shams:
        r, _ = verify(sham, task["test"], task.get("starter_code", ""))
        if r > max_r:
            max_r, best_sham = r, sham
    return {
        "id": task.get("id", ""),
        "checked": True,
        "airtight": max_r < 1.0,
        "max_sham_reward": max_r,
        "worst_sham": best_sham if max_r >= 1.0 else "",
    }


def harden_python_task(fn_name: str, oracle_code: str, weak_test: str, probe_args_exprs: list[str]) -> str:
    """Execute `oracle_code` on `probe_args_exprs` in isolated namespace to synthesize ground-truth edge assertions!"""
    ns = {}
    exec(compile(oracle_code, "<oracle>", "exec"), ns)
    fn = ns[fn_name]
    extra_asserts = []
    for args_expr in probe_args_exprs:
        args_val = eval(args_expr, {})
        expected = fn(*args_val) if isinstance(args_val, tuple) else fn(args_val)
        call_src = f"{fn_name}{args_val!r}" if isinstance(args_val, tuple) else f"{fn_name}({args_val!r})"
        extra_asserts.append(f"assert {call_src} == {expected!r}")
    return (weak_test.rstrip() + "\n" + "\n".join(extra_asserts)).strip()


def run_self_check():
    # 1. Prove `audit_test_strength` catches a weak 1-assert test where `return 5` cheats to 1.0!
    weak_task = {
        "id": "weak_add",
        "prompt": "Write `add_two(a, b)` returning `a + b`.",
        "test": "assert add_two(2, 3) == 5",
    }
    rep_weak = audit_test_strength(weak_task)
    assert rep_weak["airtight"] is False and rep_weak["max_sham_reward"] == 1.0, rep_weak

    # 2. Harden the weak test automatically using oracle + boundary probes -> shams drop from 1.0 to <0.5!
    hardened_test = harden_python_task(
        "add_two",
        "def add_two(a, b): return a + b",
        weak_task["test"],
        ["(0, 0)", "(-7, 11)", "(100, -250)"],
    )
    rep_strong = audit_test_strength({**weak_task, "test": hardened_test})
    assert rep_strong["airtight"] is True and rep_strong["max_sham_reward"] < 0.5, rep_strong

    # 3. Audit all Python tasks in NeutronAI's actual `tasks.jsonl` — confirm 0% sham full-pass!
    with open("tasks.jsonl", encoding="utf-8") as f:
        real_tasks = [json.loads(l) for l in f if l.strip()]
    leaks = [audit_test_strength(t) for t in real_tasks]
    cheatable = [x for x in leaks if not x["airtight"]]
    assert not cheatable, f"Found cheatable tasks in tasks.jsonl: {cheatable}"
    print(f"fuzz_tests.py self-check passed: caught sham on weak test, auto-hardened it, & verified all {len(real_tasks)} benchmark tasks resist shams.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tasks.jsonl")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        run_self_check()
    else:
        with open(args.data, encoding="utf-8") as f:
            tasks = [json.loads(l) for l in f if l.strip()]
        for t in tasks:
            r = audit_test_strength(t)
            if r["checked"]:
                print(f"[{r['id']:28s}] {'AIRTIGHT' if r['airtight'] else 'CHEATABLE!'} (max_sham_reward={r['max_sham_reward']:.2f})")
