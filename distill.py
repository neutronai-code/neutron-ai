"""NeutronAI Reasoning Distiller (`distill.py`) — teacher model -> execution-verified `<think>` SFT traces.

For every task, ask a strong teacher (default DeepSeek V4.1 Flash via the Onysoft gateway) to solve it, capture
its real reasoning + final answer, run the answer through `sandbox.verify`, and keep only traces that pass 100%.
Tasks the primary teacher fails get one more try with a fallback teacher (default DeepSeek V4 Pro 0813).

Output rows match `train.py --mode sft`: {"id", "teacher", "messages": [system, user, assistant]}, where the
assistant turn is `<think>\\n{reasoning}\\n</think>\\n\\n{answer}`. Results are appended as they finish, so an
interrupted run resumes where it stopped (tasks already in --out, or in --out.failed.jsonl, are skipped).

Usage (PowerShell):
  $env:ONYSOFT_API_KEY="sk-ony-..."
  uv run distill.py --limit 3 --effort low          # quick test
  uv run distill.py                                  # all tasks in train_clean.jsonl
"""
import argparse, concurrent.futures, json, os, random, threading, time, urllib.error, urllib.request
from repair import system_prompt_for
from sandbox import verify


class Deferred(Exception):
    """Temporary API trouble (rate limit, outage): the task is NOT marked failed and is retried on the next run."""


class AdaptiveLimiter:
    """Caps in-flight requests. A 429 lowers the cap by one (at most every 15s); a streak of successes raises it."""

    def __init__(self, max_concurrency: int):
        self.max = max(1, max_concurrency)
        self.limit = self.max
        self.active = 0
        self.streak = 0
        self.last_drop = 0.0
        self.cv = threading.Condition()

    def __enter__(self):
        with self.cv:
            while self.active >= self.limit:
                self.cv.wait()
            self.active += 1

    def __exit__(self, *exc):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()

    def throttled(self):
        with self.cv:
            self.streak = 0
            if self.limit > 1 and time.time() - self.last_drop > 15:
                self.limit -= 1
                self.last_drop = time.time()
                print(f"  [rate limit] slowing down to {self.limit} request(s) at a time", flush=True)

    def succeeded(self):
        with self.cv:
            self.streak += 1
            if self.limit < self.max and self.streak >= 20:
                self.limit += 1
                self.streak = 0
                print(f"  [rate limit] speeding up to {self.limit} request(s) at a time", flush=True)
                self.cv.notify_all()


LIMITER = AdaptiveLimiter(4)
_SHOWN_429 = threading.Event()


def chat(base_url: str, api_key: str, model: str, messages: list[dict], effort: str, max_tokens: int, timeout: float,
         max_wait: float = 1800.0) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if effort:
        body["reasoning_effort"] = effort
    data = json.dumps(body).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    waited, attempt = 0.0, 0
    while True:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=data, headers=headers)
        code, detail = None, ""
        with LIMITER:
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                LIMITER.succeeded()
                return resp
            except urllib.error.HTTPError as e:
                code, detail = e.code, e.read().decode("utf-8", "replace")[:400]
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                code, detail = "network", str(e)
        if code == 429:
            LIMITER.throttled()
            if not _SHOWN_429.is_set():
                _SHOWN_429.set()
                print(f"  [rate limit] first 429 from the API: {detail}", flush=True)
        elif code not in ("network", 500, 502, 503, 504):
            raise RuntimeError(f"HTTP {code}: {detail[:300]}")  # e.g. 400 bad request: a real, permanent failure
        delay = min(60.0, 5.0 * 2 ** attempt) + random.uniform(0, 3)
        attempt += 1
        waited += delay
        if waited > max_wait:
            raise Deferred(f"gave up after {int(waited)}s of {code} errors")
        time.sleep(delay)


def split_reply(resp: dict) -> tuple[str, str, str]:
    """(reasoning, answer, finish_reason) from an OpenAI-style response; reasoning field names vary by provider."""
    choice = resp["choices"][0]
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    if not reasoning and msg.get("reasoning_details"):
        reasoning = "".join(d.get("text", "") for d in msg["reasoning_details"] if isinstance(d, dict))
    return reasoning.strip(), (msg.get("content") or "").strip(), choice.get("finish_reason") or ""


def solve(task: dict, model: str, args) -> dict:
    sys_prompt = system_prompt_for(task)
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": task["prompt"]}]
    t0 = time.time()
    try:
        resp = chat(args.base_url, args.api_key, model, messages, args.effort, args.max_tokens, args.timeout)
        reasoning, answer, finish = split_reply(resp)
    except Deferred as e:
        return {"ok": False, "deferred": True, "why": str(e), "model": model}
    except Exception as e:  # permanent API error (e.g. 400): record and move on
        return {"ok": False, "why": f"api_error: {e}", "model": model}
    if not answer:
        return {"ok": False, "why": f"empty_answer (finish={finish})", "model": model}
    if not reasoning:
        return {"ok": False, "why": "no_reasoning_returned", "model": model}
    if len(reasoning) > args.max_think_chars:
        return {"ok": False, "why": f"reasoning_too_long ({len(reasoning)} chars)", "model": model}
    score, info = verify(
        answer, task["test"], task.get("starter_code") or "", task.get("files"), task.get("lang", "python"),
        perf_budget_ms=float(task.get("perf_budget_ms", 0.0) or 0.0),
    )
    if score != 1.0:
        return {"ok": False, "why": f"failed_verify {score}: {str(info.get('status'))[:120]}", "model": model}
    return {
        "ok": True,
        "model": model,
        "seconds": round(time.time() - t0, 1),
        "row": {
            "id": task.get("id", ""),
            "teacher": model,
            "messages": [
                *messages,
                {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n{answer}"},
            ],
        },
    }


def run_task(task: dict, args) -> dict:
    tries = [args.model] * args.samples + ([args.fallback_model] if args.fallback_model else [])
    last = {}
    for model in tries:
        last = solve(task, model, args)
        if last["ok"] or last.get("deferred"):  # don't burn the fallback on a rate limit
            return last
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_clean.jsonl")
    ap.add_argument("--out", default="sft_distill.jsonl")
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash:tencent")
    ap.add_argument("--fallback-model", default="deepseek/deepseek-v4-pro-0813:tencent", help="'' to disable")
    ap.add_argument("--effort", default="high", help="reasoning_effort: low | high | max ('' to omit)")
    ap.add_argument("--samples", type=int, default=2, help="attempts with --model before trying the fallback")
    ap.add_argument("--max-think-chars", type=int, default=40000, help="drop traces longer than this (~11K tokens)")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--concurrency", "--workers", type=int, default=4,
                    help="max requests in flight; drops automatically when the API says 429 (too many requests)")
    ap.add_argument("--max-wait", type=float, default=1800.0, help="seconds to keep retrying one rate-limited request")
    ap.add_argument("--shuffle", action="store_true", help="process tasks in a random (fixed-seed) order")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pending tasks (0 = all)")
    ap.add_argument("--retry-failed", action="store_true", help="also retry tasks listed in <out>.failed.jsonl")
    ap.add_argument("--base-url", default="https://api.onysoft.com/v1")
    args = ap.parse_args()
    args.api_key = os.environ.get("ONYSOFT_API_KEY", "")
    if not args.api_key:
        raise SystemExit('Set your key first:  $env:ONYSOFT_API_KEY="sk-ony-..."')

    with open(args.data, encoding="utf-8") as f:
        tasks = [json.loads(l) for l in f if l.strip()]
    failed_path = args.out + ".failed.jsonl"

    def ids_in(path, transient_ok=False):
        if not os.path.exists(path):
            return set()
        ids = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                why = row.get("why", "")
                # Older runs logged rate limits / outages as failures; those deserve another try.
                if transient_ok and why.startswith(("api_error: HTTP 429", "api_error: HTTP 5", "api_error: network")):
                    continue
                ids.add(row.get("id"))
        return ids

    skip = ids_in(args.out) | (set() if args.retry_failed else ids_in(failed_path, transient_ok=True))
    pending = [t for t in tasks if t.get("id") not in skip]
    if args.shuffle:
        random.Random(0).shuffle(pending)
    if args.limit:
        pending = pending[: args.limit]
    LIMITER.max = LIMITER.limit = max(1, args.concurrency)
    print(f"{len(tasks)} tasks | {len(skip)} already done/failed | running {len(pending)} with {args.model} "
          f"(effort={args.effort or 'default'}, fallback={args.fallback_model or 'none'}, up to {LIMITER.max} at a time)", flush=True)

    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "deferred": 0, "by_teacher": {}}
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=LIMITER.max) as pool:
        futs = {pool.submit(run_task, t, args): t for t in pending}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            task = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"ok": False, "why": f"crash: {type(e).__name__}: {e}", "model": ""}
            with lock:
                if res["ok"]:
                    stats["ok"] += 1
                    stats["by_teacher"][res["model"]] = stats["by_teacher"].get(res["model"], 0) + 1
                    with open(args.out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(res["row"]) + "\n")
                    tag = f"PASS ({res['model'].split('/')[-1]}, {res['seconds']}s)"
                elif res.get("deferred"):
                    stats["deferred"] += 1
                    tag = f"DEFERRED (retried next run): {res['why'][:70]}"
                else:
                    stats["fail"] += 1
                    with open(failed_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"id": task.get("id"), "lang": task.get("lang"), "why": res["why"]}) + "\n")
                    tag = f"FAIL {res['why'][:90]}"
                rate = i / max(1e-9, time.time() - t_start) * 3600
                print(f"[{i}/{len(pending)}] {task.get('id', '?'):40s} {tag}   (~{rate:.0f} tasks/hr)", flush=True)

    total = stats["ok"] + stats["fail"] + stats["deferred"]
    print(f"\nKept {stats['ok']}/{total} verified traces -> {args.out}   by teacher: {stats['by_teacher']}")
    if stats["fail"]:
        print(f"Failures logged to {failed_path} (rerun with --retry-failed to try them again)")
    if stats["deferred"]:
        print(f"{stats['deferred']} task(s) hit rate limits/outages; just run the same command again to finish them.")


if __name__ == "__main__":
    main()
