import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Self-training pilot, step 1 of 2 (GPU): MODEL writes a greedy fill plus N sampled fills for each pilot hole
    # (holes from 21 repos with working tests that Quark was never trained on), uploaded to the private quark-data
    # dataset as pilot/samples_<model>@<sha7>.json. Step 2 tests them on GitHub Actions ("Quark pilot tests").
    # Paste the HF token, turn the GPU on (notebook specs button at the top), click Start.
    import marimo as mo

    token = mo.ui.text(kind="password", label="Hugging Face token (hf_...)", full_width=True)
    start = mo.ui.run_button(label="Start sampling")
    mo.vstack([token, start])
    return mo, start, token


@app.cell
def _(mo, start, token):
    import json, os, shutil, signal, subprocess, sys, time, urllib.request, zipfile
    from concurrent.futures import ThreadPoolExecutor

    mo.stop(not start.value, mo.md("Paste your Hugging Face token above, then click **Start sampling**."))
    tok = token.value.strip()
    assert tok.startswith("hf_"), "Paste your Hugging Face token first"

    MODEL = "neutronai-code/Quark-12B-A2.5B-v0.7"
    N = 4  # sampled fills per hole, besides the greedy one
    TEMP = 0.8
    DATASET = "neutronai-code/quark-data"
    HOLES = "pilot/holes_ctx2.jsonl"
    work = os.path.abspath("quark_work")
    os.makedirs(work, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    # The model version, as in molab_bench.py: the newest commit that changed the model's own files
    _files = [f for f in HfApi(token=tok).list_repo_tree(MODEL, expand=True)
              if isinstance(f, RepoFile) and f.last_commit and f.path != ".gitattributes"
              and not f.path.endswith(".md")]
    SHA = (max(_files, key=lambda f: f.last_commit.date).last_commit.oid if _files
           else HfApi(token=tok).model_info(MODEL).sha)
    NAME = f"{MODEL}@{SHA[:7]}"
    OUT = f"pilot/samples_{NAME.split('/')[-1]}.json"
    print("Sampling", NAME, flush=True)

    def sh(cmd):  # run a command; show its output only if it fails
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"failed: {cmd}\n{(r.stdout + r.stderr)[-3000:]}")
        return r.stdout

    # NeutronAI's code (fim.py: the FIM prompt format), fresh every run
    code = os.path.join(work, "neutron-ai-main")
    shutil.rmtree(code, ignore_errors=True)
    zpath = os.path.join(work, "code.zip")
    urllib.request.urlretrieve("https://github.com/neutronai-code/neutron-ai/archive/refs/heads/main.zip", zpath)
    zipfile.ZipFile(zpath).extractall(work)
    sys.path.insert(0, code)
    import fim

    rows = [json.loads(l) for l in open(hf_hub_download(DATASET, HOLES, repo_type="dataset", token=tok,
                                                        local_dir=work, force_download=True), encoding="utf-8")
            if l.strip()]
    print(f"{len(rows)} pilot holes, {len({r['repo'] for r in rows})} repos", flush=True)

    with open(hf_hub_download(MODEL, "tokenizer.json", revision=SHA, token=tok), encoding="utf-8") as _f:
        _t = _f.read()
    fim._FIM_STYLE[MODEL] = fim.style_from_tokenizer(_t) or "qwen"
    print("FIM prompt format:", fim._FIM_STYLE[MODEL], flush=True)

    gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                         capture_output=True, text=True)
    assert gpu.returncode == 0 and gpu.stdout.strip(), (
        "No GPU: open the notebook specs button at the top, pick the RTX Pro 6000 GPU, Apply and restart, then run again")
    print("GPU:", gpu.stdout.strip(), flush=True)
    venv = "/tmp/vllm312"
    py = sys.executable
    if not os.path.exists(f"{venv}/bin/vllm"):
        sh(f"{py} -m pip install -q uv")
        sh(f"{py} -m uv venv {venv} --python 3.12 --clear")
        sh(f"{py} -m uv pip install --python {venv}/bin/python vllm==0.33.0")
    if not os.path.exists(f"{venv}/.kernels"):
        try:
            sh(f"{venv}/bin/vllm download-kernels")
            open(f"{venv}/.kernels", "w").close()
        except RuntimeError as e:
            print("(kernel download skipped:", str(e).strip().splitlines()[-1][:200] + ")", flush=True)

    subprocess.run("pkill -f 'vllm serve'", shell=True)
    time.sleep(5)
    free = sh("nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits").split(",")
    assert int(free[0]) > 0.86 * int(free[1]), "The GPU is still in use: restart the notebook, then run it again"
    log_path = os.path.join(work, "vllm.log")
    server = subprocess.Popen([f"{venv}/bin/vllm", "serve", MODEL, "--revision", SHA, "--tokenizer-revision", SHA,
                               "--port", "8000", "--max-model-len", "8192",
                               "--gpu-memory-utilization", "0.85", "--max-num-seqs", "128"],
                              env=dict(os.environ, HF_TOKEN=tok, VLLM_USE_FLASHINFER_SAMPLER="0"),
                              stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.time()
        while True:
            if server.poll() is not None:
                raise RuntimeError("vLLM exited while loading; the end of its log:\n" + open(log_path).read()[-3000:])
            try:
                urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
                break
            except Exception:
                if time.time() - t0 > 30 * 60:
                    raise RuntimeError("vLLM did not start in 30 min:\n" + open(log_path).read()[-3000:])
                time.sleep(10)
        print(f"vLLM serving {MODEL} after {time.time() - t0:.0f}s", flush=True)

        url, done = "http://localhost:8000/v1/completions", [0]

        def one(p):  # greedy (as in the benchmark) + N samples at TEMP
            try:
                g = fim.query_fim_samples(url, MODEL, p, max_tokens=512)
                s = fim.query_fim_samples(url, MODEL, p, max_tokens=512, temperature=TEMP, n=N)
                out = [fim.extract_fim_middle(x) for x in g + s]
            except Exception as e:
                print("request failed:", e, flush=True)
                out = [""] * (1 + N)
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"{done[0]}/{len(rows)} holes", flush=True)
            return out

        t1 = time.time()
        with ThreadPoolExecutor(32) as pool:
            fills = list(pool.map(one, [r["fim_prompt"] for r in rows]))
        exact1 = sum(f[0].strip() == r["oracle_middle"].strip() for f, r in zip(fills, rows))
        exactk = sum(any(x.strip() == r["oracle_middle"].strip() for x in f) for f, r in zip(fills, rows))
        distinct = sum(len({x.strip() for x in f if x.strip()}) for f in fills) / len(fills)
        print(f"{len(fills)} holes in {(time.time() - t1) / 60:.0f} min: greedy matches the original exactly on "
              f"{exact1}, some fill on {exactk}; {distinct:.1f} distinct fills per hole", flush=True)
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    HfApi(token=tok).upload_file(
        path_or_fileobj=json.dumps({"key": {"data": HOLES, "ids": [r["id"] for r in rows]}, "model": NAME,
                                    "temperature": TEMP, "fills": fills}).encode("utf-8"),
        path_in_repo=OUT, repo_id=DATASET, repo_type="dataset", commit_message=f"pilot samples: {NAME}")
    print(f"Samples uploaded -> {DATASET}/{OUT}\nNext: GitHub > Actions > Quark pilot tests > Run workflow", flush=True)
    return


if __name__ == "__main__":
    app.run()
