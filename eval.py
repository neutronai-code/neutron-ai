"""NeutronAI Benchmark Evaluator v0.9 (`eval.py`) — 5-Language + Speed-Gate + Multi-File Repo `pass@k` & Diffing.

Capabilities:
  - Evaluates BOTH function-synthesis tasks AND SWE-bench-style multi-file `### FILE:` SEARCH/REPLACE repo tasks.
  - Supports `--repair-turns N` to benchmark agentic test-time self-repair alongside raw 1-shot `pass@k`.
  - Supports `--compare prev_report.jsonl` to print exact per-task regressions & wins between Base and Hadron / Quark!

Examples:
  uv run eval.py --mock --n 4 --k 1,4
  uv run eval.py --url http://localhost:11434/v1/chat/completions --model qwen2.5-coder:7b --report base.jsonl
  uv run eval.py --url http://localhost:11434/v1/chat/completions --model hadron-7b --compare base.jsonl
"""
import argparse, collections, concurrent.futures, json, math, urllib.request
import repair
from repair import solve_with_repair, system_prompt_for
from sandbox import verify

MOCK_SOLUTIONS = {
    "merge_intervals": "```python\ndef merge_intervals(intervals):\n    out = []\n    for s, e in sorted(intervals):\n        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)\n        else: out.append([s, e])\n    return out\n```",
    "lru_simulation": "```python\nfrom collections import OrderedDict\ndef lru_simulation(capacity, ops):\n    c, res = OrderedDict(), []\n    for op in ops:\n        if op[0] == 'get':\n            k = op[1]\n            if capacity > 0 and k in c: c.move_to_end(k); res.append(c[k])\n            else: res.append(-1)\n        else:\n            _, k, v = op\n            if capacity <= 0: continue\n            if k in c: c.move_to_end(k)\n            c[k] = v\n            if len(c) > capacity: c.popitem(last=False)\n    return res\n```",
    "toposort_lex": "```python\nimport heapq\ndef toposort_lex(n, edges):\n    g, deg = [[] for _ in range(n)], [0]*n\n    for u, v in edges: g[u].append(v); deg[v] += 1\n    h = [i for i in range(n) if deg[i] == 0]\n    heapq.heapify(h); out = []\n    while h:\n        u = heapq.heappop(h); out.append(u)\n        for v in g[u]:\n            deg[v] -= 1\n            if deg[v] == 0: heapq.heappush(h, v)\n    return out if len(out) == n else None\n```",
    "unified_diff_added": "```python\nimport re\ndef parse_unified_line_numbers(patch):\n    ln, added = 0, []\n    for line in patch.splitlines():\n        if line.startswith('@@'): ln = int(re.search(r'\\+(\\d+)', line).group(1)) - 1\n        elif line.startswith(('+++', '---')): continue\n        elif line.startswith('+'): ln += 1; added.append(ln)\n        elif not line.startswith('-'): ln += 1\n    return sorted(added)\n```",
    "apply_search_replace": "```python\ndef apply_edits(source, edits):\n    for old, new in edits:\n        if not old or source.count(old) != 1:\n            raise ValueError('bad edit')\n        source = source.replace(old, new, 1)\n    return source\n```",
    "fix_binary_search_bug": "```python\ndef lower_bound(nums: list[int], target: int) -> int:\n    lo, hi = 0, len(nums)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if nums[mid] < target: lo = mid + 1\n        else: hi = mid\n    return lo\n```",
    "fix_mutable_default_dag": "```python\ndef collect_reachable(edges, start, seen=None):\n    if seen is None: seen = set()\n    if start in seen: return seen\n    seen.add(start)\n    for u, v in edges:\n        if u == start: collect_reachable(edges, v, seen)\n    return set(seen)\n```",
    "token_bucket_limiter": "```python\ndef simulate_rate_limiter(rate, burst, requests):\n    tokens, last, out = burst, 0.0, []\n    for t, cost in requests:\n        tokens = min(burst, tokens + (t - last) * rate)\n        last = t\n        if tokens >= cost: tokens -= cost; out.append(True)\n        else: out.append(False)\n    return out\n```",
    "json_pointer_get": "```python\ndef json_pointer(doc, pointer):\n    if pointer == '': return doc\n    if not pointer.startswith('/'): raise KeyError(pointer)\n    cur = doc\n    for raw in pointer.split('/')[1:]:\n        tok = raw.replace('~1', '/').replace('~0', '~')\n        if isinstance(cur, list):\n            if not tok.isdigit() or (len(tok) > 1 and tok.startswith('0')): raise IndexError(tok)\n            cur = cur[int(tok)]\n        elif isinstance(cur, dict): cur = cur[tok]\n        else: raise KeyError(tok)\n    return cur\n```",
    "retry_backoff": "```python\ndef retry_backoff(delays_out, fn, retries=3, base=0.5):\n    for attempt in range(retries + 1):\n        try: return fn()\n        except Exception:\n            if attempt == retries: raise\n            delays_out.append(base * (2 ** attempt))\n```",
    "minimal_indent_dedent": "```python\ndef smart_dedent(text):\n    trail = text.endswith('\\n')\n    lines = text.splitlines()\n    non_empty = [l for l in lines if l.strip()]\n    margin = min((len(l) - len(l.lstrip(' ')) for l in non_empty), default=0)\n    out = [l[margin:] if l.strip() else '' for l in lines]\n    return '\\n'.join(out) + ('\\n' if trail else '')\n```",
    "sql_like_matcher": "```python\nimport re\ndef sql_like(text, pattern):\n    rx, i = ['^'], 0\n    while i < len(pattern):\n        ch = pattern[i]\n        if ch == '\\\\' and i + 1 < len(pattern):\n            rx.append(re.escape(pattern[i + 1])); i += 2\n        elif ch == '%': rx.append('.*'); i += 1\n        elif ch == '_': rx.append('.'); i += 1\n        else: rx.append(re.escape(ch)); i += 1\n    rx.append('$')\n    return re.match(''.join(rx), text, re.DOTALL) is not None\n```",
    "repo_fix_pagination_offset": (
        "### FILE: app/math_util.py\n<<<<<<< SEARCH\ndef ceil_div(total: int, size: int) -> int:\n    return (total // size) + 1\n=======\ndef ceil_div(total: int, size: int) -> int:\n    return (total + size - 1) // size if total > 0 else 0\n>>>>>>> REPLACE\n"
        "### FILE: app/paginator.py\n<<<<<<< SEARCH\ndef paginate(items: list, page: int, size: int) -> dict:\n    start = page * size\n=======\ndef paginate(items: list, page: int, size: int) -> dict:\n    if page < 1:\n        raise ValueError('page must be >= 1')\n    start = (page - 1) * size\n>>>>>>> REPLACE"
    ),
    "repo_fix_auth_hmac_cache": (
        "### FILE: auth/tokens.py\n<<<<<<< SEARCH\n    user, exp_s, sig = parts\n    if int(exp_s) < now:\n        return None\n    return user\n=======\n    user, exp_s, sig = parts\n    expected = hmac.new(secret.encode(), f'{user}:{exp_s}'.encode(), hashlib.sha256).hexdigest()[:16]\n    if not hmac.compare_digest(sig, expected) or now >= int(exp_s):\n        return None\n    return user\n>>>>>>> REPLACE\n"
        "### FILE: auth/middleware.py\n<<<<<<< SEARCH\n    def authenticate(self, token: str, now: int) -> str | None:\n        if token in self._cache:\n            return self._cache[token]\n        user = verify_token(token, self.secret, now)\n        self._cache[token] = user\n        return user\n=======\n    def authenticate(self, token: str, now: int) -> str | None:\n        if token in self._cache:\n            user, exp = self._cache[token]\n            return user if now < exp else None\n        user = verify_token(token, self.secret, now)\n        if user is not None:\n            self._cache[token] = (user, int(token.split(':')[1]))\n        return user\n>>>>>>> REPLACE"
    ),
    "repo_fix_config_override_merge": (
        "### FILE: cfg/loader.py\n<<<<<<< SEARCH\ndef load_config(overrides: dict, env: dict[str, str]) -> dict:\n    cfg = DEFAULTS\n    cfg.update(overrides)\n    for k, v in env.items():\n        sect, sub = k.lower().split('__', 1)\n        cfg[sect][sub] = v\n    return cfg\n=======\nimport copy\n\ndef _deep_merge(base: dict, over: dict) -> dict:\n    out = copy.deepcopy(base)\n    for k, v in over.items():\n        if isinstance(v, dict) and isinstance(out.get(k), dict):\n            out[k] = _deep_merge(out[k], v)\n        else:\n            out[k] = copy.deepcopy(v)\n    return out\n\ndef load_config(overrides: dict, env: dict[str, str]) -> dict:\n    cfg = _deep_merge(DEFAULTS, overrides)\n    for k, v in env.items():\n        sect, sub = k.lower().split('__', 1)\n        orig = cfg.get(sect, {}).get(sub)\n        cfg[sect][sub] = int(v) if type(orig) is int else v\n    return cfg\n>>>>>>> REPLACE"
    ),
    "js_promise_pool_limit": (
        "```javascript\nasync function mapLimit(items, limit, worker) {\n  const ret = new Array(items.length);\n  let next = 0;\n  async function runner() {\n    while (next < items.length) {\n      const i = next++;\n      ret[i] = await worker(items[i], i);\n    }\n  }\n  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, runner));\n  return ret;\n}\n```"
    ),
    "js_deep_equal_cycles": (
        "```javascript\nfunction deepEqual(a, b, seen = new Map()) {\n  if (Object.is(a, b)) return true;\n  if (typeof a !== 'object' || a === null || typeof b !== 'object' || b === null) return false;\n  if (a instanceof Date || b instanceof Date) return a instanceof Date && b instanceof Date && a.getTime() === b.getTime();\n  if (Array.isArray(a) !== Array.isArray(b)) return false;\n  if (seen.get(a) === b) return true;\n  seen.set(a, b);\n  const ka = Object.keys(a), kb = Object.keys(b);\n  if (ka.length !== kb.length) return false;\n  for (const k of ka) {\n    if (!Object.prototype.hasOwnProperty.call(b, k) || !deepEqual(a[k], b[k], seen)) return false;\n  }\n  return true;\n}\n```"
    ),
    "js_repo_event_emitter_once": (
        "### FILE: bus/emitter.js\n<<<<<<< SEARCH\n  once(evt, fn) {\n    const wrap = (...args) => {\n      const list = this.handlers[evt] || [];\n      list.splice(list.indexOf(wrap), 1);\n      fn(...args);\n    };\n    this.on(evt, wrap);\n  }\n  emit(evt, ...args) {\n    const list = this.handlers[evt] || [];\n    for (let i = 0; i < list.length; i++) list[i](...args);\n  }\n=======\n  once(evt, fn) {\n    const unsub = this.on(evt, (...args) => {\n      unsub();\n      fn(...args);\n    });\n    return unsub;\n  }\n  emit(evt, ...args) {\n    const list = [...(this.handlers[evt] || [])];\n    for (let i = 0; i < list.length; i++) list[i](...args);\n  }\n>>>>>>> REPLACE"
    ),
    "ts_result_railway_combinator": (
        "```typescript\ntype Result<T, E> = { ok: true; value: T } | { ok: false; error: E };\nfunction ok<T>(value: T): Result<T, never> { return { ok: true, value }; }\nfunction err<E>(error: E): Result<never, E> { return { ok: false, error }; }\nfunction andThen<T, U, E>(res: Result<T, E>, fn: (val: T) => Result<U, E>): Result<U, E> {\n  return res.ok ? fn(res.value) : res;\n}\n```"
    ),
    "ts_repo_typed_router_params": (
        "### FILE: router/matcher.mts\n<<<<<<< SEARCH\n  const pSegs = pattern.split('/').filter(Boolean);\n  const uSegs = path.split('/');\n  if (pSegs.length !== uSegs.length) return null;\n  const params: Record<string, string> = {};\n  for (let i = 0; i < pSegs.length; i++) {\n    if (pSegs[i].startsWith(':')) params[pSegs[i].slice(1)] = uSegs[i];\n    else if (pSegs[i] !== uSegs[i]) return null;\n  }\n=======\n  const pSegs = pattern.split('/').filter(Boolean);\n  const uSegs = path.split('/').filter(Boolean);\n  if (pSegs.length !== uSegs.length) return null;\n  const params: Record<string, string> = {};\n  for (let i = 0; i < pSegs.length; i++) {\n    if (pSegs[i].startsWith(':')) params[pSegs[i].slice(1)] = decodeURIComponent(uSegs[i]);\n    else if (pSegs[i] !== uSegs[i]) return null;\n  }\n>>>>>>> REPLACE"
    ),
    "cpp_fenwick_range_sum": (
        "```cpp\nclass Fenwick {\n    int n_;\n    std::vector<long long> bit_;\n    long long prefix(int idx) const {\n        long long s = 0;\n        for (int i = idx + 1; i > 0; i -= i & -i) s += bit_[i];\n        return s;\n    }\npublic:\n    explicit Fenwick(int n) : n_(n), bit_(n + 1, 0) {}\n    void add(int idx, long long delta) {\n        for (int i = idx + 1; i <= n_; i += i & -i) bit_[i] += delta;\n    }\n    long long range_sum(int l, int r) const {\n        if (l > r) return 0;\n        return prefix(r) - (l > 0 ? prefix(l - 1) : 0);\n    }\n};\n```"
    ),
    "cpp_repo_ring_buffer_wrap": (
        "### FILE: ds/ring.cpp\n<<<<<<< SEARCH\nbool RingBuffer::push(int val) {\n    buf_[tail_] = val;\n    tail_ = (tail_ + 1) % cap_;\n    size_++;\n    return true;\n}\n\nstd::optional<int> RingBuffer::pop() {\n    if (size_ == 0) return std::nullopt;\n    int v = buf_[head_++];\n    size_--;\n    return v;\n}\n=======\nbool RingBuffer::push(int val) {\n    if (size_ == cap_) return false;\n    buf_[tail_] = val;\n    tail_ = (tail_ + 1) % cap_;\n    size_++;\n    return true;\n}\n\nstd::optional<int> RingBuffer::pop() {\n    if (size_ == 0) return std::nullopt;\n    int v = buf_[head_];\n    head_ = (head_ + 1) % cap_;\n    size_--;\n    return v;\n}\n>>>>>>> REPLACE"
    ),
    "sql_session_islands_cte": (
        "```sql\nWITH lagged AS (\n  SELECT user_id, ts, CASE WHEN ts - LAG(ts) OVER (PARTITION BY user_id ORDER BY ts) <= 300 THEN 0 ELSE 1 END AS is_new\n  FROM events\n), grp AS (\n  SELECT user_id, ts, SUM(is_new) OVER (PARTITION BY user_id ORDER BY ts) AS sid FROM lagged\n)\nSELECT user_id, MIN(ts) AS session_start, MAX(ts) AS session_end, COUNT(*) AS event_count\nFROM grp GROUP BY user_id, sid ORDER BY user_id, session_start\n```"
    ),
    "sql_repo_ledger_trigger_fix": (
        "### FILE: db/schema.sql\n<<<<<<< SEARCH\nCREATE TRIGGER trg_transfer AFTER INSERT ON transfers\nBEGIN\n    UPDATE accounts SET balance = balance + NEW.amount WHERE id = NEW.to_acct;\nEND;\n=======\nCREATE TRIGGER trg_transfer BEFORE INSERT ON transfers\nBEGIN\n    SELECT RAISE(ABORT, 'insufficient_funds') WHERE (SELECT balance FROM accounts WHERE id = NEW.from_acct) < NEW.amount;\n    UPDATE accounts SET balance = balance - NEW.amount WHERE id = NEW.from_acct;\n    UPDATE accounts SET balance = balance + NEW.amount WHERE id = NEW.to_acct;\nEND;\n>>>>>>> REPLACE"
    ),
    "perf_subarray_sum_count": (
        "```python\ndef count_subarrays_sum(nums: list[int], k: int) -> int:\n    seen = {0: 1}\n    cur = ans = 0\n    for x in nums:\n        cur += x\n        ans += seen.get(cur - k, 0)\n        seen[cur] = seen.get(cur, 0) + 1\n    return ans\n```"
    ),
    "perf_ts_longest_unique_window": (
        "```typescript\nfunction longestUniqueSubstr(s: string): number {\n  const last = new Map<string, number>();\n  let left = 0, best = 0;\n  for (let r = 0; r < s.length; r++) {\n    const ch = s[r];\n    const prev = last.get(ch);\n    if (prev !== undefined && prev >= left) left = prev + 1;\n    last.set(ch, r);\n    if (r - left + 1 > best) best = r - left + 1;\n  }\n  return best;\n}\n```"
    ),
}


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def query_endpoint(url: str, model: str, sys_prompt: str, prompt: str, api_key: str, temperature: float) -> str:
    payload = json.dumps({
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=repair.REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]


TEMPERATURE = None  # --temperature; None = 0.0 for n=1, 0.6 for pass@k sampling


def eval_single_task(task: dict, url: str, model: str, api_key: str, n: int, repair_turns: int, mock: bool) -> dict:
    sys_prompt = system_prompt_for(task)
    temp = TEMPERATURE if TEMPERATURE is not None else (0.0 if n == 1 else 0.6)
    passed_samples, rewards, statuses, latencies = 0, [], [], []

    for _ in range(n):
        if repair_turns > 1:
            mock_fn = (lambda t, _turn, _m: MOCK_SOLUTIONS.get(t.get("id", ""), "")) if mock else None
            rep = solve_with_repair(task, url, model, api_key, max_turns=repair_turns, mock_fn=mock_fn)
            last = rep["attempts"][-1]
            score, info = last["reward"], last["info"]
        else:
            sol = MOCK_SOLUTIONS.get(task.get("id", ""), "") if mock else query_endpoint(url, model, sys_prompt, task["prompt"], api_key, temp)
            score, info = verify(
                sol,
                task["test"],
                task.get("starter_code", ""),
                task.get("files"),
                task.get("lang", "python"),
                perf_budget_ms=float(task.get("perf_budget_ms", 0.0)),
            )
        rewards.append(score)
        statuses.append(info["status"])
        latencies.append(info.get("ms", 0.0))
        if score == 1.0:
            passed_samples += 1

    return {
        "id": task.get("id", "unknown"),
        "category": task.get("category", "general"),
        "n": n,
        "c": passed_samples,
        "pass@1": round(pass_at_k(n, passed_samples, 1), 4),
        "mean_reward": round(sum(rewards) / len(rewards), 4),
        "exec_ms": round(sum(latencies) / len(latencies), 2),
        "status": statuses[0],
    }


def print_comparison(results: list[dict], baseline_path: str):
    with open(baseline_path, encoding="utf-8") as f:
        prev = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
    gains, regressions = [], []
    for r in results:
        old = prev.get(r["id"])
        if not old:
            continue
        delta = r["pass@1"] - old.get("pass@1", pass_at_k(old["n"], old["c"], 1))
        if delta > 0.01:
            gains.append((r["id"], delta))
        elif delta < -0.01:
            regressions.append((r["id"], delta))
    print(f"\nDiff vs {baseline_path}: +{len(gains)} improved, -{len(regressions)} regressed")
    for tid, d in gains:
        print(f"  [+] {tid}: +{d:.0%}")
    for tid, d in regressions:
        print(f"  [-] {tid}: {d:.0%}")


def run_eval(data_path: str, url: str, model: str, api_key: str, n: int, ks: list[int], repair_turns: int, workers: int, report_out: str, compare_path: str, mock: bool):
    with open(data_path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(eval_single_task, t, url, model, api_key, n, repair_turns, mock) for t in tasks]
        for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            r = fut.result()
            print(f"  progress {done}/{len(tasks)}: {r['id']} -> {r['c']}/{r['n']} passed", flush=True)
        results = [f.result() for f in futs]

    by_cat = collections.defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
        print(f"[{r['category']:8s}] {r['id']:28s} :: {r['c']}/{r['n']} passed | reward={r['mean_reward']:.2f} | {r['exec_ms']:5.1f}ms ({r['status']})")

    summary = {f"pass@{k}": round(sum(pass_at_k(r["n"], r["c"], k) for r in results) / len(results), 4) for k in ks if k <= n}
    summary["mean_dense_reward"] = round(sum(r["mean_reward"] for r in results) / len(results), 4)

    print("\nCategory pass@1:")
    for cat, items in sorted(by_cat.items()):
        cat_p1 = sum(pass_at_k(r["n"], r["c"], 1) for r in items) / len(items)
        print(f"  {cat:10s}: {cat_p1:.1%} ({len(items)} tasks)")

    print("\nOverall:", " | ".join(f"{k}={v:.1%}" if k.startswith("pass@") else f"{k}={v:.3f}" for k, v in summary.items()))
    if report_out:
        with open(report_out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    if compare_path:
        print_comparison(results, compare_path)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tasks.jsonl")
    ap.add_argument("--url", default="http://localhost:11434/v1/chat/completions")
    ap.add_argument("--model", default="hadron-7b")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--k", default="1")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Sampling temperature (default 0.0 for n=1). Qwen3/3.5 thinking models: use 0.6, greedy can loop")
    ap.add_argument("--repair-turns", type=int, default=1, help="Set to 2-3 to evaluate multi-turn test-guided self-repair")
    ap.add_argument("--workers", type=int, default=6, help="Tasks sent at once; use 1 for a local CPU model (Ollama)")
    ap.add_argument("--timeout", type=float, default=180, help="Seconds to wait for each model reply; raise it (e.g. 1800) for slow local models")
    ap.add_argument("--report", default="eval_report.jsonl")
    ap.add_argument("--compare", default="", help="Previous eval_report.jsonl to diff against")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    repair.REQUEST_TIMEOUT = args.timeout
    TEMPERATURE = args.temperature
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    summary = run_eval(args.data, args.url, args.model, args.api_key, args.n, ks, args.repair_turns, args.workers, args.report, args.compare, args.mock)
    if args.mock:
        assert summary["pass@1"] == 1.0, summary
