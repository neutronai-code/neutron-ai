"""Benchmark a model on Modal's GPUs (free $30/month credits): the Modal twin of `vultr.py --job eval`.

Serves the model with vLLM (bf16), runs eval.py on each benchmark file, prints the summaries and uploads them to
the HF dataset repo under eval/<model>/. Use the SAME --gpu and settings for a "before" and an "after" run so the
two scores are comparable.

One-time setup:  uv tool install modal  ->  modal setup  ->  modal secret create huggingface HF_TOKEN=hf_...

  modal run modal_eval.py --model Qwen/Qwen3.5-4B                       # 4B baseline (L4)
  modal run modal_eval.py --model neutronai-code/Hadron-4B-v1           # trained 4B, same settings
  modal run modal_eval.py --model Qwen/Qwen3.5-9B --gpu L40S            # 9B needs a 48GB card

Default benchmarks: tasks.jsonl (26 hand-made tasks) + tasks_holdout.jsonl (cases training never sees),
4 tries per task at temperature 0.6 (pass@1 is averaged over the 4 tries, which cuts the noise a lot).
"""
import os
import pathlib
import re
import subprocess
import sys
import time

import modal

HERE = pathlib.Path(__file__).parent
CODE_FILES = ["eval.py", "repair.py", "sandbox.py"]
DATA_FILES = ["tasks.jsonl", "tasks_holdout.jsonl"]

image = (
    # Ubuntu 24.04 like the Vultr VMs: g++ 13 for -std=c++23, same OS for comparable results.
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("curl", "ca-certificates", "g++", "build-essential", "sqlite3", "git")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_24.x | bash -",
        "apt-get install -y nodejs",
    )
    .uv_pip_install("vllm==0.33.0", "huggingface_hub")
    # FlashInfer's top-k/top-p sampler JIT-compiles with nvcc, which this image lacks; vLLM's PyTorch sampler
    # draws from the same distribution.
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
for f in CODE_FILES + DATA_FILES:
    image = image.add_local_file(HERE / f, f"/app/{f}")

app = modal.App("neutron-eval", image=image)
hf_cache = modal.Volume.from_name("neutron-hf-cache", create_if_missing=True)


def _wait_for_server(proc: subprocess.Popen, minutes: int = 25):
    import urllib.request

    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("vLLM exited during startup (see its log above)")
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
            return
        except Exception:
            time.sleep(10)
    raise RuntimeError(f"vLLM did not start within {minutes} minutes")


@app.function(gpu="L4", timeout=4 * 3600, secrets=[modal.Secret.from_name("huggingface")],
              volumes={"/root/.cache/huggingface": hf_cache})
def run_eval(model: str, data_files: list[str], n: int, temperature: float, hf_repo: str) -> str:
    from huggingface_hub import HfApi

    os.chdir("/app")
    server = subprocess.Popen(
        ["vllm", "serve", model, "--port", "8000", "--max-model-len", "12288", "--gpu-memory-utilization", "0.90",
         "--max-num-seqs", "64", "--reasoning-parser", "qwen3"]
    )
    try:
        _wait_for_server(server)
        hf_cache.commit()  # keep the downloaded weights for the next run
        tag = re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1])
        api = HfApi()
        summaries = []
        for data in data_files:
            stem = pathlib.Path(data).stem
            report = f"/tmp/{stem}.jsonl"
            out = subprocess.run(
                [sys.executable, "eval.py", "--data", data, "--url", "http://localhost:8000/v1/chat/completions",
                 "--model", model, "--n", str(n), "--k", "1" if n < 4 else "1,4", "--workers", "32",
                 "--timeout", "1800", "--temperature", str(temperature), "--report", report],
                capture_output=True, text=True,
            )
            text = out.stdout + (f"\n[stderr]\n{out.stderr[-3000:]}" if out.returncode else "")
            summary = f"=== {model} on {data} (n={n}, temperature={temperature}) ===\n" + "\n".join(
                ln for ln in text.splitlines() if not ln.startswith("  progress"))
            print(summary, flush=True)
            summaries.append(summary)
            if hf_repo:
                summary_path = f"/tmp/{stem}.txt"
                pathlib.Path(summary_path).write_text(summary, encoding="utf-8")
                for local, remote in ((report, f"eval/{tag}/{stem}.jsonl"), (summary_path, f"eval/{tag}/{stem}.txt")):
                    if os.path.exists(local):
                        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=hf_repo, repo_type="dataset")
        return "\n\n".join(summaries)
    finally:
        server.terminate()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3.5-4B", gpu: str = "L4", data: str = ",".join(DATA_FILES), n: int = 4,
         temperature: float = 0.6, hf_repo: str = "neutronai-code/hadron-data"):
    files = [d.strip() for d in data.split(",") if d.strip()]
    missing = [d for d in files if d not in DATA_FILES]
    if missing:
        raise SystemExit(f"{missing} not in the image; add them to DATA_FILES in modal_eval.py")
    print(f"Benchmarking {model} on {gpu}: {files}, {n} tries per task")
    result = run_eval.with_options(gpu=gpu).remote(model, files, n, temperature, hf_repo)
    print("\n" + result)
