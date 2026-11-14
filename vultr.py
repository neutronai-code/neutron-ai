"""NeutronAI Cloud Launcher (`vultr.py`) — Zero-Waste Vultr GPU Provisioner + Auto-Self-Destruct Wrapper.

Why solo AI founders lose $300 in free cloud credits:
  - Leaving an H100/A100 instance idle overnight while sleeping (remember: Vultr `Stop` STILL BILLS YOU—only `DELETE /v2/instances/{id}` stops billing!).

What `vultr.py` solves (100% Python stdlib, zero SDK bloat):
  1. `plans`: Queries Vultr REST API v2 for live GPU plans (`vcg-*`), hourly $/hr burn rate, VRAM, and in-stock regions.
  2. `script`: Generates a battle-tested cloud-init / bash runscript (`vultr_job.sh`) that:
     - Installs `uv`, syncs NeutronAI, runs your exact `train.py` pipeline (`sft` -> `dpo` -> `grpo` / `fim-grpo`),
     - Pushes merged Hadron / Quark weights & GGUF to Hugging Face Hub (`HF_TOKEN`), and
     - **Calls `DELETE https://api.vultr.com/v2/instances/$INSTANCE_ID` inside a `trap ... EXIT` block**
       so the GPU VM **self-destructs the second training finishes OR crashes** — guaranteed $0 idle waste!
  3. `launch`: Provisions the GPU VM with that self-destruct startup script (defaults to `--dry-run` unless `--confirm` is explicitly passed so you never spend $0.01 by accident).

Usage:
  uv run vultr.py --self-check
  $env:VULTR_API_KEY="..."; uv run vultr.py plans
  uv run vultr.py script --hf-repo NeutronAI/Hadron-7B-v0.9 --steps 300 --max-minutes 60
  $env:VULTR_API_KEY="..."; $env:HF_TOKEN="..."; uv run vultr.py launch --plan vcg-a100-1c-6g-16vram --region ewr --hf-repo NeutronAI/Hadron-7B-v0.9 --dry-run
"""
import argparse, base64, json, os, urllib.request

VULTR_API = "https://api.vultr.com/v2"


def _vultr_req(method: str, path: str, api_key: str, payload: dict | None = None) -> dict:
    if not api_key:
        raise ValueError("Missing VULTR_API_KEY environment variable or --api-key")
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{VULTR_API}{path}",
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw.strip() else {}


def build_self_destruct_script(
    repo_url: str,
    base_model: str,
    dataset: str,
    steps: int,
    max_minutes: int,
    hf_repo: str,
    auto_destroy: bool = True,
) -> str:
    """Bash startup script with `trap cleanup EXIT` that destroys its own Vultr VM even if Python crashes."""
    destroy_block = (
        '  if [ -n "${VULTR_API_KEY:-}" ] && [ -n "${VULTR_INSTANCE_ID:-}" ]; then\n'
        '    echo "[NeutronAI] Triggering Vultr API self-destruct for instance ${VULTR_INSTANCE_ID}..."\n'
        '    curl -sS -X DELETE "https://api.vultr.com/v2/instances/${VULTR_INSTANCE_ID}" \\\n'
        '      -H "Authorization: Bearer ${VULTR_API_KEY}" || true\n'
        "  fi\n"
        if auto_destroy
        else '  echo "[NeutronAI] auto_destroy disabled; remember to DELETE instance manually!"\n'
    )
    hf_flag = f'--push-hf "{hf_repo}"' if hf_repo else ""
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "# Discover own Vultr instance-id via Vultr metadata service (169.254.169.254)\n"
        'VULTR_INSTANCE_ID="$(curl -sf http://169.254.169.254/v1/instance-v2-id || true)"\n\n'
        "cleanup() {\n"
        '  echo "[NeutronAI] Job exiting (code=$?). Running billing safeguard..."\n'
        f"{destroy_block}"
        "}\n"
        "trap cleanup EXIT\n\n"
        "export DEBIAN_FRONTEND=noninteractive\n"
        "curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        'export PATH="$HOME/.local/bin:$PATH"\n\n'
        "mkdir -p /workspace && cd /workspace\n"
        f'if [ -n "{repo_url}" ]; then git clone "{repo_url}" neutron-ai && cd neutron-ai; fi\n\n'
        "uv sync --extra train\n"
        f'uv run train.py --mode grpo --model "{base_model}" --data "{dataset}" '
        f"--steps {steps} --max-minutes {max_minutes} --export-gguf {hf_flag} --out hadron-out\n"
    )


def cmd_plans(api_key: str):
    body = _vultr_req("GET", "/plans?type=vcg", api_key)
    plans = body.get("plans", [])
    print(f"{'PLAN ID':32s} | {'VRAM':6s} | {'$/HR':7s} | {'$/MO':8s} | REGIONS")
    print("-" * 88)
    for p in sorted(plans, key=lambda x: float(x.get("hourly_cost", 999))):
        locs = ",".join(p.get("locations", [])[:6])
        vram = f"{p.get('gpu_vram_gb', '?')}GB"
        print(f"{p['id']:32s} | {vram:6s} | ${float(p.get('hourly_cost', 0)):<6.3f} | ${float(p.get('monthly_cost', 0)):<7.1f} | {locs}")


def cmd_launch(args) -> dict:
    script = build_self_destruct_script(
        repo_url=args.repo_url,
        base_model=args.model,
        dataset=args.data,
        steps=args.steps,
        max_minutes=args.max_minutes,
        hf_repo=args.hf_repo,
        auto_destroy=not args.keep_alive,
    )
    # Inject runtime secrets into top of cloud-init script over HTTPS only when launching
    injected = (
        f'export VULTR_API_KEY="{args.api_key}"\n'
        f'export HF_TOKEN="{args.hf_token}"\n'
    )
    full_script = script.replace("set -euo pipefail\n", f"set -euo pipefail\n{injected}", 1)
    payload = {
        "region": args.region,
        "plan": args.plan,
        "os_id": args.os_id,  # 2284 = Ubuntu 24.04 LTS x64 on Vultr
        "label": args.label,
        "hostname": "neutron-trainer",
        "user_data": base64.b64encode(full_script.encode("utf-8")).decode("ascii"),
    }
    if not args.confirm:
        print("[DRY-RUN] Pass `--confirm` to actually provision & bill Vultr. Payload preview:")
        preview = {**payload, "user_data": f"<base64 {len(payload['user_data'])} chars with trap EXIT DELETE safeguard>"}
        print(json.dumps(preview, indent=2))
        return preview

    created = _vultr_req("POST", "/instances", args.api_key, payload)
    inst = created.get("instance", {})
    print(f"Launched Vultr GPU instance {inst.get('id')} ({args.plan} in {args.region}) — auto-destroy on exit ARMED.")
    return created


def run_self_check():
    script = build_self_destruct_script(
        repo_url="https://github.com/neutron-ai/neutron.git",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        dataset="mbpp",
        steps=300,
        max_minutes=60,
        hf_repo="NeutronAI/Hadron-7B-v0.9",
    )
    assert "trap cleanup EXIT" in script, script
    assert "DELETE \"https://api.vultr.com/v2/instances/${VULTR_INSTANCE_ID}\"" in script, script
    assert "--max-minutes 60" in script and "--push-hf \"NeutronAI/Hadron-7B-v0.9\"" in script, script
    assert "169.254.169.254/v1/instance-v2-id" in script, script
    print("vultr.py (NeutronAI) self-check passed: cloud-init script verified with metadata ID discovery + trap EXIT self-destruct.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", default="script", choices=["plans", "script", "launch"])
    ap.add_argument("--api-key", default=os.environ.get("VULTR_API_KEY", ""))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--plan", default="vcg-a100-1c-6g-16vram")
    ap.add_argument("--region", default="ewr")
    ap.add_argument("--os-id", type=int, default=2284, help="Vultr OS ID (2284 = Ubuntu 24.04 x64)")
    ap.add_argument("--label", default="neutron-hadron-rlvr")
    ap.add_argument("--repo-url", default="")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--data", default="mbpp")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max-minutes", type=int, default=60)
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--out-script", default="vultr_job.sh")
    ap.add_argument("--keep-alive", action="store_true", help="Disable automatic VM DELETE on exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm", action="store_true", help="Required to actually create a billable Vultr VM")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        run_self_check()
    elif args.action == "plans":
        cmd_plans(args.api_key)
    elif args.action == "launch":
        cmd_launch(args)
    else:
        s = build_self_destruct_script(args.repo_url, args.model, args.data, args.steps, args.max_minutes, args.hf_repo, not args.keep_alive)
        with open(args.out_script, "w", encoding="utf-8", newline="\n") as f:
            f.write(s)
        print(f"Wrote self-destructing Vultr GPU job script -> {args.out_script}")
