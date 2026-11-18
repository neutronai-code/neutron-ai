"""Quark base-model bake-off on Modal: which small model fills in code best?

Each candidate is served with vLLM and gets the raw fill-in-the-middle prompt
(`<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>`) for every task in fim_holdout.jsonl (mined from held-out cases
by `fim.py mine`). A completion counts only if the spliced file passes the case's real tests.

  modal run modal_fim_eval.py
  modal run modal_fim_eval.py --models Qwen/Qwen2.5-Coder-1.5B,Qwen/Qwen3.5-2B-Base --gpu L4
  modal run modal_fim_eval.py --models ... --data fim_holdout_hard.jsonl      # the longer-hole test set

Difficulty filter for the next Quark round (writes fim_train_v2.jsonl here and to the HF dataset repo):
  modal run modal_fim_eval.py --filter --models neutronai-code/Quark-1.5B-v0.1
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import modal

HERE = pathlib.Path(__file__).parent
FILES = ["fim.py", "sandbox.py", "repomap.py", "fim_holdout.jsonl", "fim_holdout_hard.jsonl"]
POOL = ["fim_train.jsonl", "fim_hard_pool.jsonl"]  # holes the filter scores (only baked into the filter's image)
DEFAULT_MODELS = "Qwen/Qwen2.5-Coder-1.5B,Qwen/Qwen3.5-0.8B-Base,Qwen/Qwen3.5-2B-Base"

image = (  # same base layers as modal_eval.py, so Modal reuses the cached build
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("curl", "ca-certificates", "g++", "build-essential", "sqlite3", "git")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_24.x | bash -",
        "apt-get install -y nodejs",
    )
    .uv_pip_install("vllm==0.33.0", "huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
for f in FILES:
    if (HERE / f).exists():
        image = image.add_local_file(HERE / f, f"/app/{f}")
filter_image = image
for f in POOL:
    if (HERE / f).exists():
        filter_image = filter_image.add_local_file(HERE / f, f"/app/{f}")

app = modal.App("neutron-fim-bakeoff", image=image)
hf_cache = modal.Volume.from_name("neutron-hf-cache", create_if_missing=True)


def _serve(model: str) -> subprocess.Popen:
    import urllib.request

    cmd = ["vllm", "serve", model, "--port", "8000", "--max-model-len", "8192", "--gpu-memory-utilization", "0.85",
           "--max-num-seqs", "64"]
    if re.search(r"qwen3[._]5", model, re.I):  # text only: skip the image/video encoder's big memory profile
        cmd += ["--limit-mm-per-prompt", '{"image": 0, "video": 0}']
    # Own process group, so stopping it also kills vLLM's engine subprocess and frees the GPU for the next model.
    proc = subprocess.Popen(cmd, start_new_session=True)
    deadline = time.time() + 25 * 60
    while time.time() < deadline:
        if proc.poll() is not None:
            _stop(proc)
            raise RuntimeError(f"vLLM exited while loading {model}")
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
            return proc
        except Exception:
            time.sleep(10)
    _stop(proc)
    raise RuntimeError(f"vLLM did not start for {model}")


def _stop(proc: subprocess.Popen):
    """Kill the whole vLLM process group, then wait until the GPU memory is actually released."""
    import signal

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=60)
            break
        except (ProcessLookupError, subprocess.TimeoutExpired):
            continue
    for _ in range(30):
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip().splitlines()
        if used and int(used[0]) < 1500:
            return
        time.sleep(2)


@app.function(gpu="A10G", timeout=3 * 3600, secrets=[modal.Secret.from_name("huggingface")],
              volumes={"/root/.cache/huggingface": hf_cache})
def bakeoff(models: list[str], hf_repo: str, data: str = "fim_holdout.jsonl") -> str:
    from huggingface_hub import HfApi

    os.chdir("/app")
    sys.path.insert(0, "/app")
    import fim

    # Original test set keeps its old upload path so earlier results stay comparable; others get a subfolder.
    folder = "fim_bakeoff" if data == "fim_holdout.jsonl" else f"fim_bakeoff/{pathlib.Path(data).stem}"
    lines = []
    for model in models:
        tag = re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1])
        try:
            server = _serve(model)
        except RuntimeError as e:
            lines.append(f"{model}: FAILED TO LOAD ({e})")
            print(lines[-1], flush=True)
            continue
        try:
            hf_cache.commit()
            report = f"/tmp/fim_{tag}.jsonl"
            score = fim.eval_fim(data, "http://localhost:8000/v1/completions", model, "", "completions",
                                 workers=24, limit=0, report=report)
            lines.append(f"{model}: FIM pass@1 {score:.1%}")
            if hf_repo:
                HfApi().upload_file(path_or_fileobj=report, path_in_repo=f"{folder}/{tag}.jsonl", repo_id=hf_repo,
                                    repo_type="dataset")
        finally:
            _stop(server)
    summary = "\n".join(lines)
    print(f"\n=== Quark bake-off ({data}) ===\n" + summary, flush=True)
    return summary


@app.function(gpu="A10G", cpu=8, timeout=4 * 3600, secrets=[modal.Secret.from_name("huggingface")],
              volumes={"/root/.cache/huggingface": hf_cache}, image=filter_image)
def filter_holes(model: str, samples: int, hf_repo: str, out_name: str) -> str:
    from huggingface_hub import HfApi

    os.chdir("/app")
    sys.path.insert(0, "/app")
    import fim

    seen, pool = set(), []
    for f in POOL:
        if os.path.exists(f):
            for line in open(f, encoding="utf-8"):
                if line.strip() and (task_id := json.loads(line)["id"]) not in seen:
                    seen.add(task_id)
                    pool.append(line if line.endswith("\n") else line + "\n")
    with open("/tmp/pool.jsonl", "w", encoding="utf-8") as fo:
        fo.writelines(pool)
    print(f"Scoring {len(pool)} holes x {samples} answers with {model}", flush=True)
    server = _serve(model)
    try:
        hf_cache.commit()
        fim.filter_fim("/tmp/pool.jsonl", f"/tmp/{out_name}", "http://localhost:8000/v1/completions", model,
                       samples=samples, temperature=0.8, workers=24)
    finally:
        _stop(server)
    if hf_repo:
        HfApi().upload_file(path_or_fileobj=f"/tmp/{out_name}", path_in_repo=f"fim/{out_name}", repo_id=hf_repo,
                            repo_type="dataset")
    return open(f"/tmp/{out_name}", encoding="utf-8").read()


@app.local_entrypoint()
def main(models: str = DEFAULT_MODELS, gpu: str = "A10G", hf_repo: str = "neutronai-code/hadron-data",
         data: str = "fim_holdout.jsonl", filter: bool = False, samples: int = 4, out: str = "fim_train_v2.jsonl"):
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if filter:  # difficulty filter for the next training round, scored by ONE model (the one being trained further)
        missing = [f for f in POOL if not (HERE / f).exists()]
        if missing:
            raise SystemExit(f"Missing {missing}. Mine the longer holes first: uv run fim.py mine --hard --source "
                             "tasks_generated.jsonl --out fim_hard_pool.jsonl --holdout-out fim_holdout_hard.jsonl")
        if len(model_list) != 1:
            raise SystemExit("--filter needs exactly one model, e.g. --models neutronai-code/Quark-1.5B-v0.1")
        text = filter_holes.with_options(gpu=gpu).remote(model_list[0], samples, hf_repo, out)
        (HERE / out).write_text(text, encoding="utf-8")
        n = sum(1 for line in text.splitlines() if line.strip())
        print(f"\nWrote {n} training holes -> {out}. Train the next Quark on it with --steps {n}.")
        return
    n_tasks = sum(1 for line in open(HERE / data, encoding="utf-8") if line.strip())
    print(f"FIM bake-off on {gpu}: {len(model_list)} models x {n_tasks} held-out fill-in tasks ({data})")
    print("\n" + bakeoff.with_options(gpu=gpu).remote(model_list, hf_repo, data))
