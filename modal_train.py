"""Run train.py on a Modal GPU (pay per second from the free monthly credits): used for Quark, while Vultr trains Hadron.

The image installs the project's exact locked dependencies (uv.lock, `train` extra) like the Vultr VMs do, plus g++ and
Node 24 for the test-running reward, and cmake for the GGUF export. Checkpoints go to <--push-hf>/last-checkpoint every
50 steps; if Modal restarts the container, the retry resumes from there.

  modal run modal_train.py                                    # Quark: FIM-GRPO on Qwen2.5-Coder-1.5B
  modal run modal_train.py --steps 200 --max-minutes 60       # shorter run
"""
import os
import pathlib
import subprocess
import sys

import modal

HERE = pathlib.Path(__file__).parent
CODE = ["train.py", "sandbox.py", "fim.py", "repomap.py", "repair.py", "hf_processor.py"]

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("curl", "ca-certificates", "g++", "build-essential", "sqlite3", "git", "cmake",
                 "libcurl4-openssl-dev")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_24.x | bash -",
        "apt-get install -y nodejs",
    )
    .uv_sync(str(HERE), extras=["train"])
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

app = modal.App("neutron-train", image=image)
hf_cache = modal.Volume.from_name("neutron-hf-cache", create_if_missing=True)


def _with_files(img: modal.Image, data: str) -> modal.Image:
    for f in CODE + [data]:
        img = img.add_local_file(HERE / f, f"/app/{f}")
    return img


@app.function(gpu="A10G", timeout=24 * 3600, secrets=[modal.Secret.from_name("huggingface")],
              volumes={"/root/.cache/huggingface": hf_cache}, retries=modal.Retries(max_retries=2),
              image=_with_files(image, "fim_train.jsonl"))
def train(args: list[str]) -> int:
    os.chdir("/app")
    cmd = [sys.executable, "train.py", *args]
    print("[NeutronAI] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-Coder-1.5B", mode: str = "fim-grpo", data: str = "fim_train.jsonl",
         steps: int = 1500, max_minutes: int = 600, max_seq_len: int = 1536,
         push_hf: str = "neutronai-code/Quark-1.5B-v0.1", gpu: str = "A10G", extra: str = ""):
    if data != "fim_train.jsonl":
        raise SystemExit("Only fim_train.jsonl is baked into the image; edit modal_train.py to add another file.")
    args = ["--mode", mode, "--model", model, "--data", data, "--steps", str(steps), "--max-minutes", str(max_minutes),
            "--max-seq-len", str(max_seq_len), "--load-in-4bit", "no", "--push-hf", push_hf, "--out", "/tmp/run-out",
            "--resume", "--export-gguf", *extra.split()]
    print(f"Training {model} ({mode}) on Modal {gpu}: {steps} steps, max {max_minutes} min -> {push_hf}")
    code = train.with_options(gpu=gpu).remote(args)
    print(f"\ntrain.py exited with code {code}")
