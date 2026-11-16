"""NeutronAI Data Engine (`synthesize.py`): measure task difficulty with the base model + harvest verified traces.

For every task the model answers `--samples` times. Each answer is run against the task's tests, giving:
  * rl_curriculum.jsonl - tasks the model does NOT always solve, tagged with `baseline_pass_rate`
                          (train.py puts the ~50% ones first: that is where RL learns most)
  * sft_verified.jsonl  - the shortest correct answer per task (with its <think> reasoning), for SFT

Every finished task is appended to `--log` immediately, so a crash or timeout loses nothing: rerun with the same
`--log` to resume, or `--finalize-only` to rebuild the two outputs from the log.

Local (Ollama):   uv run synthesize.py --data tasks_generated.jsonl --model qwen3.5:4b --workers 1 --timeout 1800
GPU (vLLM):       uv run synthesize.py --data tasks_generated.jsonl --url http://localhost:8000/v1/chat/completions \
                    --model Qwen/Qwen3.5-4B --workers 32 --max-tokens 6144 --thinking on
"""
import argparse, collections, concurrent.futures, json, os, threading, time, urllib.error, urllib.request
from repair import system_prompt_for
from sandbox import verify


def sample_completion(url: str, model: str, sys_prompt: str, prompt: str, api_key: str, temperature: float,
                      max_tokens: int = 0, timeout: float = 180, thinking: bool | None = None) -> str:
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    if thinking is not None:  # vLLM / SGLang: Qwen3/3.5 thinking switch
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        msg = json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]
    content = msg.get("content") or ""
    # Servers with a reasoning parser return the thinking separately; put it back so SFT traces keep it.
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if reasoning and "</think>" not in content:
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}"
    return content


def _sample_with_retry(*a, retries: int = 2, **kw) -> str | None:
    for attempt in range(retries + 1):
        try:
            return sample_completion(*a, **kw)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as e:
            if attempt == retries:
                print(f"  request failed: {e}", flush=True)
                return None
            time.sleep(5 * (attempt + 1))


def _passes(c: str, t: dict) -> bool:
    try:
        return verify(c, t["test"], t.get("starter_code", ""), t.get("files"), t.get("lang", "python"),
                      perf_budget_ms=float(t.get("perf_budget_ms", 0.0)))[0] == 1.0
    except Exception:
        return False


def _task_key(t: dict, i: int) -> str:
    return t.get("id") or f"task_{i}"


def finalize(tasks: list[dict], log_path: str, sft_out: str, rl_out: str) -> tuple[int, int]:
    """Build sft_out / rl_out from the per-task log (last entry per task wins)."""
    results = {}
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results[r["key"]] = r
    sft_rows, rl_rows = [], []
    rates_by_lang = collections.defaultdict(list)
    for i, t in enumerate(tasks):
        r = results.get(_task_key(t, i))
        if r is None:
            continue
        rates_by_lang[t.get("lang", "python")].append(r["pass_rate"])
        if r.get("best"):
            sft_rows.append({
                "id": t.get("id", ""),
                "messages": [
                    {"role": "system", "content": system_prompt_for(t)},
                    {"role": "user", "content": t["prompt"]},
                    {"role": "assistant", "content": r["best"]},
                ],
            })
        if not (r["n"] >= 4 and r["pass_rate"] == 1.0):
            rl_rows.append({**t, "baseline_pass_rate": r["pass_rate"]})
    with open(sft_out, "w", encoding="utf-8") as f:
        for row in sft_rows:
            f.write(json.dumps(row) + "\n")
    with open(rl_out, "w", encoding="utf-8") as f:
        for row in rl_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n{'lang':14s} {'tasks':>6s} {'always':>7s} {'never':>6s} {'sometimes':>10s}")
    for lang, rates in sorted(rates_by_lang.items()):
        always = sum(r == 1.0 for r in rates)
        never = sum(r == 0.0 for r in rates)
        print(f"{lang:14s} {len(rates):6d} {always:7d} {never:6d} {len(rates) - always - never:10d}")
    print(f"\nWrote {len(sft_rows)} verified SFT traces -> {sft_out} | {len(rl_rows)} RL curriculum tasks -> {rl_out}")
    return len(sft_rows), len(rl_rows)


def run_engine(tasks_path: str, sft_out: str, rl_out: str, url: str, model: str, api_key: str, samples: int, mock: bool,
               workers: int = 1, max_tokens: int = 0, timeout: float = 180, thinking: bool | None = None,
               log_path: str = "", limit: int = 0, max_minutes: float = 0.0, temperature: float = 0.7):
    with open(tasks_path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    if limit:
        tasks = tasks[:limit]
    log_path = log_path or rl_out + ".log.jsonl"
    if mock and os.path.exists(log_path):
        os.remove(log_path)
    done = set()
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
        print(f"Resuming: {len(done)} tasks already in {log_path}", flush=True)
    todo = [(i, t) for i, t in enumerate(tasks) if _task_key(t, i) not in done]
    deadline = time.time() + max_minutes * 60 if max_minutes else 0.0
    lock, counter = threading.Lock(), [len(done)]

    def work(i: int, t: dict):
        if deadline and time.time() > deadline:
            return
        if mock:
            from eval import MOCK_SOLUTIONS
            good = MOCK_SOLUTIONS.get(t["id"], "")
            candidates = [f"<think>Solve {t['id']} cleanly.</think>\n{good}", "```python\ndef broken(): pass\n```"]
        else:
            sys_prompt = system_prompt_for(t)
            candidates = [_sample_with_retry(url, model, sys_prompt, t["prompt"], api_key, temperature,
                                             max_tokens=max_tokens, timeout=timeout, thinking=thinking)
                          for _ in range(samples)]
            if any(c is None for c in candidates):  # server trouble: don't record a fake 0%, retry on resume
                return
        wins = [c for c in candidates if _passes(c, t)]
        rate = round(len(wins) / len(candidates), 3) if candidates else 0.0
        rec = {"key": _task_key(t, i), "n": len(candidates), "pass_rate": rate, "best": min(wins, key=len) if wins else ""}
        with lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            counter[0] += 1
            print(f"[{counter[0]}/{len(tasks)}] {rec['key'][:40]:40s} {len(wins)}/{len(candidates)}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(lambda it: work(*it), todo))
    return finalize(tasks, log_path, sft_out, rl_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tasks.jsonl")
    ap.add_argument("--sft-out", default="sft_verified.jsonl")
    ap.add_argument("--rl-out", default="rl_curriculum.jsonl")
    ap.add_argument("--log", default="", help="Per-task results, appended as they finish (default: <rl-out>.log.jsonl)")
    ap.add_argument("--url", default="http://localhost:11434/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1, help="Tasks in flight at once (1 for Ollama on CPU, ~32 for vLLM)")
    ap.add_argument("--max-tokens", type=int, default=0, help="Cap per answer (thinking included); 0 = server default")
    ap.add_argument("--timeout", type=float, default=180, help="Seconds to wait for each answer")
    ap.add_argument("--thinking", choices=["default", "on", "off"], default="default",
                    help="Send Qwen3/3.5 enable_thinking (vLLM/SGLang only)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0.0, help="Stop starting new tasks after this long, then write outputs")
    ap.add_argument("--finalize-only", action="store_true", help="Just rebuild the outputs from --log")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    if args.finalize_only:
        with open(args.data, encoding="utf-8") as f:
            all_tasks = [json.loads(line) for line in f if line.strip()]
        finalize(all_tasks, args.log or args.rl_out + ".log.jsonl", args.sft_out, args.rl_out)
        raise SystemExit(0)
    think = {"default": None, "on": True, "off": False}[args.thinking]
    sft_n, rl_n = run_engine(args.data, args.sft_out, args.rl_out, args.url, args.model, args.api_key, args.samples,
                             args.mock, workers=args.workers, max_tokens=args.max_tokens, timeout=args.timeout,
                             thinking=think, log_path=args.log, limit=args.limit, max_minutes=args.max_minutes,
                             temperature=args.temperature)
    if args.mock:
        assert sft_n == 26 and rl_n == 26, (sft_n, rl_n)
