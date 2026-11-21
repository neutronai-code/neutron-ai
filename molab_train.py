import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Quark (Mellum2) training on molab's free GPU (https://molab.marimo.io): continues from the
    # Hugging Face checkpoint in REPO, pushes a checkpoint every 100 steps and the merged model at the end.
    # Turn the GPU on first (notebook specs button at the top), paste your HF token, click Start.
    # The token lives only in this session's memory: it is never saved into the notebook.
    import marimo as mo

    token = mo.ui.text(kind="password", label="Hugging Face token (hf_...)", full_width=True)
    start = mo.ui.run_button(label="Start training")
    mo.vstack([token, start])
    return mo, start, token


@app.cell
def _(mo, start, token):
    import os, re, shutil, subprocess, sys, time, urllib.request, zipfile

    import torch

    mo.stop(not start.value, mo.md("Paste your Hugging Face token above, then click **Start training**."))
    assert token.value.strip().startswith("hf_"), "Paste your Hugging Face token first"
    assert torch.cuda.is_available(), "No GPU: turn it on with the notebook specs button at the top, then rerun"

    # LoRA stage 1 (v0.6): Mellum2 base -> one pass over fim_v05 (59.4k rows / 16 rows per step).
    # LoRA stage 2 (v0.7): v0.6's merged model -> one pass over fim_v06 (120k new holes, other seed).
    # Full fine-tune test (the active run): every weight trains (the MoE router stays frozen), with the same data and
    # steps as stage 1, so its score compares directly with v0.6's LoRA (59.8%).
    # Each run resumes from its own REPO/last-checkpoint, so rerunning after a shutdown continues where it was.
    # MODEL, DATA, STEPS, REPO, FULL = "JetBrains/Mellum2-12B-A2.5B-Base", "fim_v05.jsonl", 3714, "neutronai-code/Quark-12B-A2.5B-v0.6", False
    # MODEL, DATA, STEPS, REPO, FULL = "neutronai-code/Quark-12B-A2.5B-v0.6", "fim_v06.jsonl", 7456, "neutronai-code/Quark-12B-A2.5B-v0.7", False
    MODEL, DATA, STEPS, REPO, FULL = "JetBrains/Mellum2-12B-A2.5B-Base", "fim_v05.jsonl", 3714, "neutronai-code/Quark-12B-A2.5B-v0.8-full", True
    # SMOKE = True: a ~20-minute trial that uploads nothing. It trains 2 steps (the longest batch comes first, so this
    # is the memory test), saves a checkpoint, resumes it for 2 more steps and prints the peak GPU memory. Run it
    # once before a new kind of run, then set SMOKE = False for the real one.
    SMOKE = True
    # Stop training after 11 h 20 min (LoRA) / 10 h 30 min (full): setup takes ~15 min (full: ~25, it downloads its
    # ~50 GB checkpoint), and the end of a session saves and uploads a checkpoint (full: ~50 GB), so this leaves time
    # before molab's 12 h session limit. A hard stop loses at most one checkpoint interval.
    MAX_MIN = 630 if FULL else 680
    work = os.path.abspath("quark_work")
    os.makedirs(work, exist_ok=True)

    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print(f"free disk: {shutil.disk_usage(work).free / 1e9:.0f} GB (the run needs about 60 GB)", flush=True)

    # 1. Libraries: the versions the Modal runs used (molab already has torch)
    # (full fine-tune: torchao for 8-bit AdamW with bf16 stochastic rounding; it works with any recent torch)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "peft==0.21.3",
                    "datasets", "accelerate", *(["torchao==0.18.0"] if FULL else [])], check=True)

    # 2. NeutronAI's code (public GitHub repo)
    code = os.path.join(work, "neutron-ai-main")  # fresh every run, so train.py is always the newest
    shutil.rmtree(code, ignore_errors=True)
    zpath = os.path.join(work, "code.zip")
    urllib.request.urlretrieve("https://github.com/neutronai-code/neutron-ai/archive/refs/heads/main.zip", zpath)
    zipfile.ZipFile(zpath).extractall(work)

    # 3. Training data (private HF dataset)
    from huggingface_hub import hf_hub_download

    data = hf_hub_download("neutronai-code/quark-data", DATA, repo_type="dataset",
                           token=token.value.strip(), local_dir=work)

    # 4. Does this GPU have the fast mixture-of-experts kernel? (just a report: train.py runs either way)
    try:
        _a = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
        _b = torch.randn(2, 32, 48, device="cuda", dtype=torch.bfloat16)
        torch._grouped_mm(_a, _b, offs=torch.tensor([32, 64], device="cuda", dtype=torch.int32))
        print("grouped_mm kernel: works on this GPU", flush=True)
    except Exception as e:
        print("grouped_mm kernel test failed (training may be slow or fail):", e, flush=True)

    # 5. Train: resumes from REPO/last-checkpoint; prints losses, notes, errors and progress every 2 minutes
    env = dict(os.environ, HF_TOKEN=token.value.strip(), PYTHONUNBUFFERED="1")
    keep = re.compile(r"NeutronAI|BudgetGuard|'loss'|train_runtime|Finished|Traceback|Error\b|Exception|"
                      r"Killed|out of memory|OutOfMemory")
    log_path = os.path.join(work, "train.log")

    def train(steps, out, push, save_steps, max_min):
        args = [sys.executable, "-u", "train.py", "--mode", "fim-sft", "--model", MODEL,
                "--data", data, "--steps", str(steps), "--max-minutes", str(max_min), "--max-seq-len", "4096",
                "--load-in-4bit", "no", "--out", out, "--resume", "--batch-size", "2", "--grad-accum", "8",
                "--save-steps", str(save_steps), "--lr", "2e-5" if FULL else "1e-4",
                *(["--full-finetune"] if FULL else []), *(["--push-hf", push] if push else [])]
        bar = re.compile(r"\d+/%d \[" % steps)
        proc = subprocess.Popen(args, cwd=code, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        last_bar, buf = 0.0, b""
        try:
            with open(log_path, "ab") as log:
                for chunk in iter(lambda: proc.stdout.read1(65536), b""):
                    log.write(chunk)
                    log.flush()
                    *lines, buf = re.split(rb"[\r\n]", buf + chunk)
                    for raw in lines:
                        line = raw.decode("utf-8", "replace").strip()
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
        shutil.rmtree(smoke, ignore_errors=True)
        print("\nSmoke test passed: set SMOKE = False and run again for the real training." if ok else
              "\nSmoke test FAILED: see the error above.", flush=True)
    else:
        train(STEPS, os.path.join(work, "run"), REPO, 300 if FULL else 100, MAX_MIN)
    return


if __name__ == "__main__":
    app.run()
