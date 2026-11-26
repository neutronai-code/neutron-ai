import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Hadron chat SFT on molab's free GPU (https://molab.marimo.io): JetBrains Mellum2.1 + a LoRA adapter,
    # trained on the verified traces from distill_agent.py. This is separate from molab_train.py (Quark FIM).
    #
    # Before starting: turn the GPU on in notebook specs, upload sft_distill_mellum_clean.jsonl, and upload this
    # repo's updated train.py while the plain-Mellum SFT change is not yet on the public GitHub main branch.
    # The JSONL is uploaded straight to this session: it is NOT published to Hugging Face.
    # SMOKE = True below trains 2 steps, saves a checkpoint, then resumes it for 2 more, uploading nothing.
    import marimo as mo

    token = mo.ui.text(kind="password", label="Hugging Face token (hf_...)", full_width=True)
    data_file = mo.ui.file(label="Training data: sft_distill_mellum_clean.jsonl", filetypes=[".jsonl"], kind="button")
    code_file = mo.ui.file(label="Updated train.py (optional once the SFT change is on GitHub main)",
                           filetypes=[".py"], kind="button")
    start = mo.ui.run_button(label="Start training")
    mo.vstack([token, data_file, code_file, start])
    return mo, start, token, data_file, code_file


@app.cell
def _(mo, start, token, data_file, code_file):
    import json
    import math
    import os
    import re
    import shutil
    import subprocess
    import sys
    import time
    import urllib.request
    import zipfile

    import torch

    mo.stop(not start.value, mo.md("Paste your HF token, upload the clean JSONL and updated `train.py`, then click Start."))
    assert token.value.strip().startswith("hf_"), "Paste your Hugging Face token first"
    assert torch.cuda.is_available(), "No GPU: turn it on with the notebook specs button at the top, then rerun"
    assert data_file.value, "Upload sft_distill_mellum_clean.jsonl first"

    MODEL = "JetBrains/Mellum2.1-12B-A2.5B-Thinking"
    REPO = "neutronai-code/Hadron-Mellum2.1-12B-A2.5B-sft1"  # merged model + checkpoints; smoke uploads nothing
    EPOCHS = 2
    SMOKE = True  # run this first; after it passes, set False and rerun for the real adapter
    MAX_MIN = 180  # wall-clock guard for the full run, in minutes
    work = os.path.abspath("hadron_sft_work")
    os.makedirs(work, exist_ok=True)

    raw = data_file.contents()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    assert rows and all(isinstance(r.get("messages"), list) for r in rows), "This is not an SFT messages JSONL"
    steps = math.ceil(len(rows) * EPOCHS / 8)  # train.py SFT: micro-batch 2 x gradient accumulation 4
    data = os.path.join(work, "sft_distill_mellum_clean.jsonl")
    with open(data, "wb") as f:
        f.write(raw)
    print(f"{len(rows)} SFT rows | {EPOCHS} epochs = {steps} optimizer steps", flush=True)
    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)

    # 1. The versions pinned by uv.lock. TRL is required for train.py --mode sft (Quark FIM doesn't need it).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "peft==0.21.3",
                    "trl==1.15.0", "datasets", "accelerate"], check=True)

    # 2. NeutronAI's public code. Uploading train.py above overlays the local version, so an unpushed SFT change works.
    code = os.path.join(work, "neutron-ai-main")
    shutil.rmtree(code, ignore_errors=True)
    zpath = os.path.join(work, "code.zip")
    urllib.request.urlretrieve("https://github.com/neutronai-code/neutron-ai/archive/refs/heads/main.zip", zpath)
    zipfile.ZipFile(zpath).extractall(work)
    if code_file.value:
        source = code_file.contents().decode("utf-8")
        assert "def load_plain_lora" in source and 'elif args.mode == "sft"' in source, "That file is not this repo's train.py"
        with open(os.path.join(code, "train.py"), "w", encoding="utf-8") as f:
            f.write(source)
        print("Using the uploaded train.py", flush=True)
    else:
        print("Using train.py from GitHub main (upload the updated file if main lacks plain-Mellum SFT)", flush=True)

    # 3. Train. Checkpoints go to REPO during the real run, and train.py uploads the merged model at the end.
    # Smoke writes local files only.
    env = dict(os.environ, HF_TOKEN=token.value.strip(), PYTHONUNBUFFERED="1")
    keep = re.compile(r"NeutronAI|BudgetGuard|'loss'|train_runtime|Finished|Traceback|Error\b|Exception|"
                      r"Killed|out of memory|OutOfMemory")
    log_path = os.path.join(work, "train.log")

    def train(n_steps, out, push, save_steps, max_min):
        args = [sys.executable, "-u", "train.py", "--mode", "sft", "--model", MODEL,
                "--data", data, "--steps", str(n_steps), "--max-minutes", str(max_min),
                "--max-seq-len", "3072", "--load-in-4bit", "no", "--out", out, "--resume",
                "--save-steps", str(save_steps), *(["--push-hf", push] if push else [])]
        bar = re.compile(rf"\d+/{n_steps} \[")
        proc = subprocess.Popen(args, cwd=code, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        last_bar, buf = 0.0, b""
        try:
            with open(log_path, "ab") as log:
                for chunk in iter(lambda: proc.stdout.read1(65536), b""):
                    log.write(chunk)
                    log.flush()
                    *lines, buf = re.split(rb"[\r\n]", buf + chunk)
                    for raw_line in lines:
                        line = raw_line.decode("utf-8", "replace").strip()
                        if bar.search(line):
                            if time.time() - last_bar > 120:
                                print(line, flush=True)
                                last_bar = time.time()
                        elif line and keep.search(line):
                            print(line, flush=True)
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
        print(f"train.py exited with code {proc.returncode} (full log: {log_path})", flush=True)
        if proc.returncode:
            with open(log_path, "rb") as f:
                print(b"\n".join(re.split(rb"[\r\n]+", f.read())[-40:]).decode("utf-8", "replace"))
        return proc.returncode

    if SMOKE:
        smoke = os.path.join(work, "smoke")
        shutil.rmtree(smoke, ignore_errors=True)
        print("== Smoke test, part 1: 2 steps, then a checkpoint", flush=True)
        ok = train(2, smoke, "", 2, 0) == 0
        print("== Smoke test, part 2: resume that checkpoint, 2 more steps", flush=True)
        ok = ok and train(4, smoke, "", 2, 0) == 0
        print("\nSmoke test passed: set SMOKE = False and run again for the real adapter." if ok else
              "\nSmoke test FAILED: see the error above.", flush=True)
    else:
        train(steps, os.path.join(work, "run"), REPO, 25, MAX_MIN)


if __name__ == "__main__":
    app.run()
