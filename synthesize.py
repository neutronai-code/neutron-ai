"""NeutronAI Data Engine v0.9 (`synthesize.py`) — Single-File + Multi-File Repo Trajectory Synthesizer."""
import argparse, concurrent.futures, json, urllib.request
from repair import system_prompt_for
from sandbox import verify


def sample_completion(url: str, model: str, sys_prompt: str, prompt: str, api_key: str, temperature: float) -> str:
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
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]


def run_engine(tasks_path: str, sft_out: str, rl_out: str, url: str, model: str, api_key: str, samples: int, mock: bool):
    with open(tasks_path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    sft_rows, rl_rows = [], []
    for t in tasks:
        sys_prompt = system_prompt_for(t)
        if mock:
            from eval import MOCK_SOLUTIONS
            good = MOCK_SOLUTIONS.get(t["id"], "")
            candidates = [f"<think>Solve {t['id']} cleanly.</think>\n{good}", "```python\ndef broken(): pass\n```"]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(samples, 8)) as pool:
                futs = [pool.submit(sample_completion, url, model, sys_prompt, t["prompt"], api_key, 0.7) for _ in range(samples)]
                candidates = [f.result() for f in futs]

        wins = [
            c for c in candidates
            if verify(
                c,
                t["test"],
                t.get("starter_code", ""),
                t.get("files"),
                t.get("lang", "python"),
                perf_budget_ms=float(t.get("perf_budget_ms", 0.0)),
            )[0] == 1.0
        ]
        pass_rate = len(wins) / len(candidates) if candidates else 0.0
        if wins:
            best = min(wins, key=len)
            sft_rows.append({
                "id": t.get("id", ""),
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": t["prompt"]},
                    {"role": "assistant", "content": best},
                ],
            })
        if not (len(candidates) >= 4 and pass_rate == 1.0):
            rl_rows.append({**t, "baseline_pass_rate": round(pass_rate, 3)})
        print(f"[{t.get('id', 'task'):28s}] verified {len(wins)}/{len(candidates)} ({pass_rate:.0%})")

    with open(sft_out, "w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r) + "\n")
    with open(rl_out, "w", encoding="utf-8") as f:
        for r in rl_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(sft_rows)} verified SFT traces -> {sft_out} | {len(rl_rows)} RL curriculum tasks -> {rl_out}")
    return len(sft_rows), len(rl_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tasks.jsonl")
    ap.add_argument("--sft-out", default="sft_verified.jsonl")
    ap.add_argument("--rl-out", default="rl_curriculum.jsonl")
    ap.add_argument("--url", default="http://localhost:11434/v1/chat/completions")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    sft_n, rl_n = run_engine(args.data, args.sft_out, args.rl_out, args.url, args.model, args.api_key, args.samples, args.mock)
    if args.mock:
        assert sft_n == 26 and rl_n == 26, (sft_n, rl_n)
