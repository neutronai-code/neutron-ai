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
  $env:VULTR_API_KEY="..."; uv run vultr.py status   # your VMs + IPs (password is printed once, at launch)
  uv run vultr.py script --hf-repo NeutronAI/Hadron-7B-v0.9 --steps 300 --max-minutes 60
  $env:VULTR_API_KEY="..."; $env:HF_TOKEN="..."; uv run vultr.py launch --plan vcg-a100-1c-6g-16vram --region ewr --hf-repo NeutronAI/Hadron-7B-v0.9 --dry-run
"""
import argparse, base64, json, os, urllib.error, urllib.request

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
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vultr API HTTP {e.code} on {method} {path}: {detail}") from None


def build_self_destruct_script(
    repo_url: str,
    base_model: str,
    dataset: str,
    steps: int,
    max_minutes: int,
    hf_repo: str,
    auto_destroy: bool = True,
) -> str:
    """Bash startup script with `trap cleanup EXIT` that uploads run logs & destroys its own Vultr VM even on crash."""
    log_upload_block = (
        f'  if [ -n "${{HF_TOKEN:-}}" ] && [ -f /var/log/neutron-train.log ]; then\n'
        f'    python3 -c "import os, urllib.request; '\
        f"t=os.environ['HF_TOKEN']; "\
        f"urllib.request.urlopen(urllib.request.Request('https://huggingface.co/api/repos/create', data=b'{{\\\"name\\\":\\\"{hf_repo.split('/')[-1]}\\\"' + (b',\\\"organization\\\":\\\"{hf_repo.split('/')[0]}\\\"' if '/' in '{hf_repo}' else b'') + b'}}', headers={{'Authorization':f'Bearer {{t}}','Content-Type':'application/json'}}))"
        f'" >/dev/null 2>&1 || true\n'
        f'    "$HOME/.local/bin/uvx" --from huggingface_hub hf upload "{hf_repo}" /var/log/neutron-train.log neutron-train.log --token "$HF_TOKEN" >/dev/null 2>&1 || true\n'
        f"  fi\n"
        if hf_repo
        else ""
    )
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
        "exec > >(tee -a /var/log/neutron-train.log) 2>&1\n"
        "set -euo pipefail\n"
        'export HOME="${HOME:-/root}"\n'
        "export DEBIAN_FRONTEND=noninteractive\n\n"
        "# Discover own Vultr instance-id via Vultr metadata service (169.254.169.254)\n"
        'VULTR_INSTANCE_ID="$(curl -sf http://169.254.169.254/v1/instance-v2-id || curl -sf http://169.254.169.254/v1.json | python3 -c \'import sys,json; print(json.load(sys.stdin).get(\"instance-v2-id\",\"\"))\' || true)"\n\n'
        "cleanup() {\n"
        '  echo "[NeutronAI] Job exiting (code=$?). Uploading log & running billing safeguard..."\n'
        f"{log_upload_block}"
        f"{destroy_block}"
        "}\n"
        "trap cleanup EXIT\n\n"
        # Fail before the slow apt/driver installs: without a repo there is nothing to train.
        f'if [ -z "{repo_url}" ]; then echo "[NeutronAI] ERROR: no repo_url baked into this script; nothing to train."; exit 1; fi\n\n'
        "apt-get -o DPkg::Lock::Timeout=180 update -qq\n"
        "apt-get -o DPkg::Lock::Timeout=180 install -y -qq git build-essential g++ sqlite3 curl ca-certificates python3\n"
        "if ! command -v nvidia-smi >/dev/null 2>&1; then\n"
        '  echo "[NeutronAI] Installing NVIDIA headless server driver..."\n'
        "  apt-get -o DPkg::Lock::Timeout=180 install -y -qq ubuntu-drivers-common && ubuntu-drivers install --gpgpu || true\n"
        "  modprobe nvidia || true\n"
        "fi\n"
        "curl -fsSL https://deb.nodesource.com/setup_24.x | bash -\n"
        "apt-get -o DPkg::Lock::Timeout=180 install -y -qq nodejs\n"
        "curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        'export PATH="$HOME/.local/bin:$PATH"\n\n'
        "nvidia-smi\n"
        "mkdir -p /workspace && cd /workspace\n"
        f'git clone "{repo_url}" neutron-ai && cd neutron-ai\n\n'
        "uv sync --extra train\n"
        f'uv run train.py --mode grpo --model "{base_model}" --data "{dataset}" '
        f"--steps {steps} --max-minutes {max_minutes} --export-gguf {hf_flag} --out hadron-out\n"
    )


def cmd_plans(api_key: str):
    # Query ALL Vultr plans + filter to any GPU plan (`vcg-*`, `vgpu`, or GPU VRAM > 0) + bare-metal GPUs (`vbm-*`)!
    all_plans = _vultr_req("GET", "/plans?per_page=500", api_key).get("plans", [])
    gpu_plans = [
        p for p in all_plans
        if p.get("id", "").startswith("vcg") or p.get("gpu_vram_gb") or p.get("gpu_type")
    ]
    try:
        bm_plans = _vultr_req("GET", "/plans-metal", api_key).get("plans_metal", [])
        for bp in bm_plans:
            if bp.get("gpu_vram_gb") or bp.get("gpu_type") or "gpu" in str(bp).lower() or any(k in bp.get("id", "") for k in ("a100", "h100", "l40", "gh200", "mi300", "a40")):
                gpu_plans.append({**bp, "_bare_metal": True})
    except Exception:
        pass

    print(f"{'PLAN ID':36s} | {'TYPE':5s} | {'VRAM':6s} | {'$/HR':7s} | IN-STOCK REGIONS")
    print("-" * 96)
    for p in sorted(gpu_plans, key=lambda x: float(x.get("hourly_cost", 999))):
        pid = p["id"]
        locs_list = p.get("locations") or []
        locs = ",".join(locs_list[:8]) if locs_list else "(sold out / account-locked)"
        vram = f"{p.get('gpu_vram_gb') or p.get('vram') or '?'}GB"
        kind = "METAL" if p.get("_bare_metal") else "VM"
        print(f"{pid:36s} | {kind:5s} | {vram:6s} | ${float(p.get('hourly_cost', 0)):<6.3f} | {locs}")


def cmd_launch(args) -> dict:
    if not args.repo_url:
        raise SystemExit("--repo-url is required for launch: the VM clones it to get train.py and the dataset.")
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
        "label": args.label,
        "hostname": "neutron-trainer",
        "user_data": base64.b64encode(full_script.encode("utf-8")).decode("ascii"),
    }
    if args.image_id:
        payload["image_id"] = args.image_id
    else:
        payload["os_id"] = args.os_id  # 2284 = Ubuntu 24.04 LTS x64 on Vultr
    if not args.confirm:
        print("[DRY-RUN] Pass `--confirm` to actually provision & bill Vultr. Payload preview:")
        preview = {**payload, "user_data": f"<base64 {len(payload['user_data'])} chars with trap EXIT DELETE safeguard>"}
        print(json.dumps(preview, indent=2))
        return preview

    created = _vultr_req("POST", "/instances", args.api_key, payload)
    inst = created.get("instance", {})
    print(f"Launched Vultr GPU instance {inst.get('id')} ({args.plan} in {args.region}) - auto-destroy on exit ARMED.")
    # Vultr returns the root password ONLY in this create response, so show it now.
    if inst.get("default_password"):
        print(f"  root password : {inst['default_password']}   (shown once - copy it now)")
    ip = inst.get("main_ip", "")
    print(f"  IP address    : {ip if ip and ip != '0.0.0.0' else 'not assigned yet - run `uv run vultr.py status` in ~1 min'}")
    return created


def cmd_status(api_key: str):
    """List your instances with IP + state, so you can `ssh root@<ip>` without opening the dashboard."""
    insts = _vultr_req("GET", "/instances?per_page=100", api_key).get("instances", [])
    if not insts:
        print("No Vultr instances (nothing is billing).")
        return
    print(f"{'LABEL':22s} | {'IP':15s} | {'STATUS':8s} | {'SERVER':12s} | {'PLAN':26s} | ID")
    print("-" * 120)
    for i in insts:
        ip = i.get("main_ip") or "-"
        print(f"{i.get('label', ''):22s} | {(ip if ip != '0.0.0.0' else 'pending'):15s} | {i.get('status', ''):8s} | "
              f"{i.get('server_status', ''):12s} | {i.get('plan', ''):26s} | {i.get('id', '')}")
        if i.get("default_password"):
            print(f"{'':22s}   root password: {i['default_password']}")
    ready = [i for i in insts if i.get("main_ip") and i["main_ip"] != "0.0.0.0"]
    if ready:
        print(f'\nWatch training:  ssh root@{ready[0]["main_ip"]} "tail -f /var/log/neutron-train.log"')


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
    ap.add_argument("action", nargs="?", default="script", choices=["plans", "script", "launch", "status"])
    ap.add_argument("--api-key", default=os.environ.get("VULTR_API_KEY", ""))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--plan", default="vcg-a100-1c-6g-16vram")
    ap.add_argument("--region", default="ewr")
    ap.add_argument("--os-id", type=int, default=2284, help="Vultr OS ID (2284 = Ubuntu 24.04 x64; Vultr auto-installs GPU drivers on vcg-* plans)")
    ap.add_argument("--image-id", default="", help="Optional Vultr Marketplace image_id")
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
    elif args.action == "status":
        cmd_status(args.api_key)
    elif args.action == "launch":
        cmd_launch(args)
    else:
        s = build_self_destruct_script(args.repo_url, args.model, args.data, args.steps, args.max_minutes, args.hf_repo, not args.keep_alive)
        with open(args.out_script, "w", encoding="utf-8", newline="\n") as f:
            f.write(s)
        print(f"Wrote self-destructing Vultr GPU job script -> {args.out_script}")
