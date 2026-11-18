"""Run train.py on a Modal GPU (pay per second from the free monthly credits): used for Quark, while Vultr trains Hadron.

The image installs the project's exact locked dependencies (uv.lock, `train` extra) like the Vultr VMs do, plus g++ and
Node 24 for the test-running reward, and cmake for the GGUF export. Checkpoints go to <--push-hf>/last-checkpoint every
50 steps; if Modal restarts the container, the retry resumes from there.

  modal run modal_train.py                                    # Quark: FIM-GRPO on Qwen2.5-Coder-1.5B
  modal run modal_train.py --steps 200 --max-minutes 60       # shorter run
  modal run modal_train.py --model neutronai-code/Quark-1.5B-v0.1 --data fim_train_v2.jsonl \
      --push-hf neutronai-code/Quark-1.5B-v0.2 --steps <holes>   # next round: continue from v0.1 on harder holes
  modal run modal_train.py --mode fim-sft --model neutronai-code/Quark-1.5B-v0.2 --data fim_sft_v03.jsonl \
      --max-seq-len 2048 --push-hf neutronai-code/Quark-1.5B-v0.3-sft --steps <rows / 16>   # v0.3 warm-up (SFT)
  modal deploy modal_repo_fim.py                              # once: the test-running reward containers
  modal run modal_train.py --mode repo-grpo --model neutronai-code/Quark-1.5B-v0.3-sft --data repo_grpo.jsonl \
      --push-hf neutronai-code/Quark-1.5B-v0.3 --steps 100   # v0.3 GRPO: reward = the repo's own tests pass
  modal run modal_train.py --mode fim-sft --model neutronai-code/Quark-1.5B-v0.3-sft --data fim_v04.jsonl \
      --gpu A100 --max-seq-len 3072 --steps 7450 --max-minutes 600 --push-hf neutronai-code/Quark-1.5B-v0.4 \
      --extra "--full-finetune"   # v0.4: full fine-tune on quark_data_v04.py's 119k rows (one pass = rows / 16)
  modal run --detach modal_train.py --mode fim-sft --model neutronai-code/Quark-1.5B-v0.4 --data fim_v05.jsonl \
      --gpu A100 --max-seq-len 4096 --steps <rows / 16> --max-minutes 600 --push-hf neutronai-code/Quark-1.5B-v0.5 \
      --extra "--full-finetune --lr 1e-5 --batch-size 2 --grad-accum 8"   # v0.5: ctx_gather.py context in 70% of holes
"""
import os
import pathlib
import subprocess
import sys

import modal

HERE = pathlib.Path(__file__).parent
CODE = ["train.py", "sandbox.py", "fim.py", "repomap.py", "repair.py", "hf_processor.py", "repo_fim.py"]

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
# The Modal secret holding HF_TOKEN. Another Modal account may already use "huggingface" for something else:
# create e.g. `modal secret create neutron-hf HF_TOKEN=hf_...` there and run with NEUTRON_HF_SECRET=neutron-hf.
HF_SECRET = os.environ.get("NEUTRON_HF_SECRET", "huggingface")


DATA = [f for f in ("fim_train.jsonl", "fim_train_v2.jsonl", "fim_sft_v03.jsonl", "repo_grpo.jsonl",
                    "fim_v04_smoke.jsonl", "fim_v04.jsonl.gz", "fim_v05.jsonl.gz")
        if (HERE / f).exists()]  # baked in when present. Big files go up gzipped (fim_v04: 505MB -> 112MB)
                                 # and are unpacked in the container; pass --data without the .gz.


def _with_files(img: modal.Image, data: list[str]) -> modal.Image:
    for f in CODE + data:
        img = img.add_local_file(HERE / f, f"/app/{f}")
    return img


@app.function(gpu="A10G", timeout=24 * 3600, secrets=[modal.Secret.from_name(HF_SECRET)],
              volumes={"/root/.cache/huggingface": hf_cache}, retries=modal.Retries(max_retries=2),
              image=_with_files(image, DATA))
def train(args: list[str]) -> int:
    os.chdir("/app")
    for f in os.listdir("/app"):  # unpack gzipped data files (uploaded small, read as plain .jsonl)
        if f.endswith(".jsonl.gz") and not os.path.exists(f[:-3]):
            import gzip
            import shutil

            with gzip.open(f, "rb") as src, open(f[:-3], "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            print(f"[NeutronAI] unpacked {f}", flush=True)
    cmd = [sys.executable, "train.py", *args]
    print("[NeutronAI] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-Coder-1.5B", mode: str = "fim-grpo", data: str = "fim_train.jsonl",
         steps: int = 1500, max_minutes: int = 600, max_seq_len: int = 1536,
         push_hf: str = "neutronai-code/Quark-1.5B-v0.1", gpu: str = "A10G", extra: str = ""):
    if data not in DATA and f"{data}.gz" not in DATA:
        raise SystemExit(f"{data} is not baked into the image (have: {DATA}); edit DATA in modal_train.py.")
    if mode == "repo-grpo":
        try:  # the reward needs the deployed test containers; fail here, not after the GPU has loaded the model
            modal.Cls.from_name("neutron-repo-fim", "RepoChecker").hydrate()
        except Exception as e:
            raise SystemExit(f"Deploy the reward containers first: modal deploy modal_repo_fim.py ({e})")
        if max_seq_len == 1536:  # default: 1,280-token prompts + 512-token fills
            max_seq_len = 1792
        extra = f"--max-prompt-tokens 1280 {extra}"
    args = ["--mode", mode, "--model", model, "--data", data, "--steps", str(steps), "--max-minutes", str(max_minutes),
            "--max-seq-len", str(max_seq_len), "--load-in-4bit", "no", "--push-hf", push_hf, "--out", "/tmp/run-out",
            "--resume", *(["--export-gguf"] if push_hf else []), *extra.split()]  # --push-hf "": a dry run
    print(f"Training {model} ({mode}) on Modal {gpu}: {steps} steps, max {max_minutes} min -> {push_hf}")
    code = train.with_options(gpu=gpu).remote(args)
    print(f"\ntrain.py exited with code {code}")
