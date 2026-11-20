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

    # Stage 1 (v0.6): Mellum2 base -> one pass over fim_v05 (59.4k rows / 16 rows per step).
    # Stage 2 (v0.7): v0.6's merged model -> one pass over fim_v06 (120k new holes, other seed). Swap the comments.
    # Each stage resumes from its own REPO/last-checkpoint, so rerunning after a shutdown continues where it was.
    # MODEL, DATA, STEPS, REPO = "JetBrains/Mellum2-12B-A2.5B-Base", "fim_v05.jsonl", 3714, "neutronai-code/Quark-12B-A2.5B-v0.6"
    MODEL, DATA, STEPS, REPO = "neutronai-code/Quark-12B-A2.5B-v0.6", "fim_v06.jsonl", 7456, "neutronai-code/Quark-12B-A2.5B-v0.7"
    # Stop training after 11 h 20 min: setup takes ~15 min and the final save + merge + upload ~5 min, so this leaves
    # ~20 min spare before molab's 12 h session limit. (A hard stop would still lose at most 100 steps: every
    # checkpoint goes to REPO/last-checkpoint.)
    MAX_MIN = 680
    work = os.path.abspath("quark_work")
    os.makedirs(work, exist_ok=True)

    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print(f"free disk: {shutil.disk_usage(work).free / 1e9:.0f} GB (the run needs about 60 GB)", flush=True)

    # 1. Libraries: the versions the Modal runs used (molab already has torch)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "peft==0.21.3",
                    "datasets", "accelerate"], check=True)

    # 2. NeutronAI's code (public GitHub repo)
    code = os.path.join(work, "neutron-ai-main")
    if not os.path.isdir(code):
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
    args = [sys.executable, "-u", "train.py", "--mode", "fim-sft", "--model", MODEL,
            "--data", data, "--steps", str(STEPS), "--max-minutes", str(MAX_MIN), "--max-seq-len", "4096",
            "--load-in-4bit", "no", "--push-hf", REPO, "--out", os.path.join(work, "run"), "--resume",
            "--lr", "1e-4", "--batch-size", "2", "--grad-accum", "8", "--save-steps", "100"]
    env = dict(os.environ, HF_TOKEN=token.value.strip(), PYTHONUNBUFFERED="1")
    keep = re.compile(r"NeutronAI|BudgetGuard|'loss'|train_runtime|Finished|Traceback|Error\b|Exception|"
                      r"Killed|out of memory|OutOfMemory")
    bar = re.compile(r"\d+/%d \[" % STEPS)
    log_path = os.path.join(work, "train.log")
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
    print(f"train.py exited with code {proc.returncode} (full log: {log_path})")
    if proc.returncode:
        with open(log_path, "rb") as f:
            print(b"\n".join(re.split(rb"[\r\n]+", f.read())[-40:]).decode("utf-8", "replace"))
    return


if __name__ == "__main__":
    app.run()
