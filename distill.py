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
import argparse, concurrent.futures, json, os, threading, time, urllib.error, urllib.request
from repair import system_prompt_for
from sandbox import verify


def chat(base_url: str, api_key: str, model: str, messages: list[dict], effort: str, max_tokens: int, timeout: float) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2 ** attempt * 3)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 4:
                time.sleep(2 ** attempt * 3)
                continue
            raise RuntimeError(f"network: {e}") from None
    raise RuntimeError("unreachable")


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
    except Exception as e:  # network/API failure: record and move on
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
        if last["ok"]:
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
    ap.add_argument("--workers", type=int, default=8)
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

    def ids_in(path):
        if not os.path.exists(path):
            return set()
        with open(path, encoding="utf-8") as f:
            return {json.loads(l).get("id") for l in f if l.strip()}

    skip = ids_in(args.out) | (set() if args.retry_failed else ids_in(failed_path))
    pending = [t for t in tasks if t.get("id") not in skip]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(tasks)} tasks | {len(skip)} already done/failed | running {len(pending)} with {args.model} "
          f"(effort={args.effort or 'default'}, fallback={args.fallback_model or 'none'})", flush=True)

    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "by_teacher": {}}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
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
                else:
                    stats["fail"] += 1
                    with open(failed_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"id": task.get("id"), "lang": task.get("lang"), "why": res["why"]}) + "\n")
                    tag = f"FAIL {res['why'][:90]}"
                print(f"[{i}/{len(pending)}] {task.get('id', '?'):40s} {tag}", flush=True)

    total = stats["ok"] + stats["fail"]
    print(f"\nKept {stats['ok']}/{total} verified traces -> {args.out}   by teacher: {stats['by_teacher']}")
    if stats["fail"]:
        print(f"Failures logged to {failed_path} (rerun with --retry-failed to try them again)")


if __name__ == "__main__":
    main()
