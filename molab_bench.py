import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Quark benchmark fills on molab's free GPU: serves MODEL with vLLM 0.33 (the Modal benchmark's version, so the
    # fills match how the other models were scored), writes its greedy fill for each of the 600 benchmark holes, and
    # uploads them to the private quark-data dataset. The repo tests then run on Modal (modal_repo_fim.py --eval
    # reuses these fills instead of a GPU). Turn the GPU on, paste the HF token, click Start.
    import marimo as mo

    token = mo.ui.text(kind="password", label="Hugging Face token (hf_...)", full_width=True)
    start = mo.ui.run_button(label="Start benchmark fills")
    mo.vstack([token, start])
    return mo, start, token


@app.cell
def _(mo, start, token):
    import json, os, signal, subprocess, sys, time, urllib.request, zipfile
    from concurrent.futures import ThreadPoolExecutor

    mo.stop(not start.value, mo.md("Paste your Hugging Face token above, then click **Start benchmark fills**."))
    tok = token.value.strip()
    assert tok.startswith("hf_"), "Paste your Hugging Face token first"

    MODEL = "neutronai-code/Quark-12B-A2.5B-v0.6"
    BENCH = "bench/repo_holes_ctx2.jsonl"  # the 600 holes with editor-style (names) context, 4k chars
    OUT = f"bench/fills_{MODEL.split('/')[-1]}_ctx2.json"
    work = os.path.abspath("quark_work")
    os.makedirs(work, exist_ok=True)

    def sh(cmd):  # run a command; show its output only if it fails (marimo hides subprocess output)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"failed: {cmd}\n{(r.stdout + r.stderr)[-3000:]}")
        return r.stdout

    # 1. NeutronAI's code (fim.py: the FIM prompt format, stop strings and fill cleanup the Modal benchmark uses)
    code = os.path.join(work, "neutron-ai-main")
    if not os.path.isdir(code):
        zpath = os.path.join(work, "code.zip")
        urllib.request.urlretrieve("https://github.com/neutronai-code/neutron-ai/archive/refs/heads/main.zip", zpath)
        zipfile.ZipFile(zpath).extractall(work)
    sys.path.insert(0, code)
    import fim

    # The FIM format comes from the model's tokenizer.json: a fine-tune saved by transformers 5 has no special-token
    # list in tokenizer_config.json, which the older fim.fim_style read, so v0.6 was sent Qwen-format prompts.
    from huggingface_hub import hf_hub_download as _dl

    with open(_dl(MODEL, "tokenizer.json", token=tok), encoding="utf-8") as _f:
        _t = _f.read()
    fim._FIM_STYLE[MODEL] = "starcoder" if '"<fim_prefix>"' in _t else "granite" if '"<|filename|>"' in _t else "qwen"
    print("FIM prompt format:", fim._FIM_STYLE[MODEL], "(Mellum must be starcoder)", flush=True)

    # 2. vLLM 0.33 on Python 3.12 in its own environment (molab's Python 3.13 can't run it), plus the precompiled
    # kernels for this GPU. vLLM 0.33 has no vllm._C module any more: the check is whether it detects the GPU.
    gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                         capture_output=True, text=True)
    assert gpu.returncode == 0 and gpu.stdout.strip(), ("No GPU attached: open the notebook specs button at the top, "
                                                        "pick the RTX Pro 6000 GPU, Apply and restart, then run again")
    print("GPU:", gpu.stdout.strip(), flush=True)
    venv = "/tmp/vllm312"
    py = sys.executable
    if not os.path.exists(f"{venv}/bin/vllm"):
        sh(f"{py} -m pip install -q uv")
        sh(f"{py} -m uv venv {venv} --python 3.12 --clear")
        sh(f"{py} -m uv pip install --python {venv}/bin/python vllm==0.33.0")
    if not os.path.exists(f"{venv}/.kernels"):  # optional speed-up: precompiled FlashInfer kernels for this GPU
        try:
            sh(f"{venv}/bin/vllm download-kernels")
            open(f"{venv}/.kernels", "w").close()
        except RuntimeError as e:
            print("(kernel download skipped:", str(e).strip().splitlines()[-1][:200] + ")", flush=True)
    print(sh(f"{venv}/bin/python -c \"from vllm.platforms import current_platform as p; import vllm; "
             f"assert p.device_type == 'cuda', p.device_type; print('vLLM', vllm.__version__, 'sees the GPU')\"")
          .strip().splitlines()[-1], flush=True)

    # 3. The benchmark holes
    from huggingface_hub import HfApi, hf_hub_download

    rows = [json.loads(l) for l in open(hf_hub_download("neutronai-code/quark-data", BENCH, repo_type="dataset",
                                                        token=tok, local_dir=work), encoding="utf-8") if l.strip()]
    print(f"{len(rows)} benchmark holes", flush=True)

    # 4. Serve the model (same settings as modal_fim_eval._serve / the Modal benchmark image). vLLM wants 85% of the
    # GPU, so first stop a vLLM left over from an earlier run; a model loaded in this notebook's own memory needs
    # a notebook restart instead.
    subprocess.run("pkill -f 'vllm serve'", shell=True)
    time.sleep(5)
    free = sh("nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits").split(",")
    print(f"GPU memory free: {int(free[0]) / 1024:.0f} of {int(free[1]) / 1024:.0f} GiB", flush=True)
    assert int(free[0]) > 0.86 * int(free[1]), ("The GPU is still in use (e.g. a model from an earlier run in this "
                                                "notebook): restart the notebook, then run it again")
    log_path = os.path.join(work, "vllm.log")
    server = subprocess.Popen([f"{venv}/bin/vllm", "serve", MODEL, "--port", "8000", "--max-model-len", "8192",
                               "--gpu-memory-utilization", "0.85", "--max-num-seqs", "64"],
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
                    raise RuntimeError("vLLM did not start in 30 min; the end of its log:\n" + open(log_path).read()[-3000:])
                time.sleep(10)
        print(f"vLLM serving {MODEL} after {time.time() - t0:.0f}s", flush=True)

        # 5. Greedy fills, exactly like modal_repo_fim.generate
        url, done = "http://localhost:8000/v1/completions", [0]

        def one(p):
            try:
                out = fim.extract_fim_middle(fim.query_fim_samples(url, MODEL, p, max_tokens=512)[0])
            except Exception as e:
                print("request failed:", e, flush=True)
                out = ""
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"{done[0]}/{len(rows)} fills", flush=True)
            return out

        print("sample fill:\n" + one(rows[0]["fim_prompt"])[:300] + "\n--- original:\n" + rows[0]["oracle_middle"][:300],
              flush=True)
        done[0] = 0
        t1 = time.time()
        with ThreadPoolExecutor(32) as pool:
            fills = list(pool.map(one, [r["fim_prompt"] for r in rows]))
        exact = sum(f.strip() == r["oracle_middle"].strip() for f, r in zip(fills, rows))
        med = sorted(len(f) for f in fills)[len(fills) // 2]
        print(f"{len(fills)} fills in {time.time() - t1:.0f}s: {exact} match the original exactly (untrained Mellum2: "
              f"165); median {med} chars (original code: ~98); {sum(not f for f in fills)} empty", flush=True)
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:  # already exited
            pass

    # 6. Upload in modal_repo_fim's fills-cache format
    blob = json.dumps({"key": {"data": "repo_holes_ctx2.jsonl", "ids": [r["id"] for r in rows]},
                       "fills": {MODEL: fills}}).encode("utf-8")
    HfApi(token=tok).upload_file(path_or_fileobj=blob, path_in_repo=OUT, repo_id="neutronai-code/quark-data",
                                 repo_type="dataset", commit_message=f"benchmark fills (vLLM): {MODEL}")
    print(f"Uploaded -> neutronai-code/quark-data/{OUT}. Done: tell Claude to run the tests.")
    return


if __name__ == "__main__":
    app.run()
