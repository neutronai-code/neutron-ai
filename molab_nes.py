import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Next-edit benchmark on molab's GPU: vLLM runs each model on the next-edit rows of the held-out repos
    # (nes_data.py --heldout -> quark-data/bench/nes_bench.jsonl), scores the rewrites (nes_eval.py) and uploads them
    # to quark-data/bench/nes_preds_<model>@<sha7>.json. No tests to run: the score is printed here.
    # Paste the HF token, turn the GPU on (notebook specs button at the top), click Start.
    import marimo as mo

    token = mo.ui.text(kind="password", label="Hugging Face token (hf_...)", full_width=True)
    start = mo.ui.run_button(label="Start next-edit benchmark")
    mo.vstack([token, start])
    return mo, start, token


@app.cell
def _(mo, start, token):
    import json, os, shutil, signal, subprocess, sys, time, urllib.request, zipfile
    from concurrent.futures import ThreadPoolExecutor

    mo.stop(not start.value, mo.md("Paste your Hugging Face token above, then click **Start next-edit benchmark**."))
    tok = token.value.strip()
    assert tok.startswith("hf_"), "Paste your Hugging Face token first"

    MODELS = [
        "neutronai-code/Quark-12B-A2.5B-v0.7",  # never trained on next edits: the starting point
        "neutronai-code/Quark-12B-A2.5B-v0.8-nes",  # v0.7 + the next-edit LoRA (molab_train.py)
    ]
    REDO = False  # True: run models again even if their predictions are already uploaded
    DATASET = "neutronai-code/quark-data"
    BENCH = "bench/nes_bench.jsonl"
    work = os.path.abspath("quark_work")
    os.makedirs(work, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.hf_api import RepoFile

    api = HfApi(token=tok)

    def version(model):  # as in ci_bench.py: the newest commit that changed the model's own (top-folder) files
        files = [f for f in api.list_repo_tree(model, expand=True)
                 if isinstance(f, RepoFile) and f.last_commit and f.path != ".gitattributes"
                 and not f.path.endswith(".md")]
        return max(files, key=lambda f: f.last_commit.date).last_commit.oid if files else api.model_info(model).sha

    def sh(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"failed: {cmd}\n{(r.stdout + r.stderr)[-3000:]}")
        return r.stdout

    def exists(name):
        try:
            hf_hub_download(DATASET, name, repo_type="dataset", token=tok, local_dir=work, force_download=True)
            return True
        except Exception as e:
            if "404" in str(e) or "EntryNotFound" in type(e).__name__:
                return False
            raise

    # NeutronAI's code (nes_eval.py: the request and the scores), fresh every run
    code = os.path.join(work, "neutron-ai-main")
    shutil.rmtree(code, ignore_errors=True)
    zpath = os.path.join(work, "code.zip")
    urllib.request.urlretrieve("https://github.com/neutronai-code/neutron-ai/archive/refs/heads/main.zip", zpath)
    zipfile.ZipFile(zpath).extractall(work)
    sys.path.insert(0, code)
    import nes_eval

    rows = [json.loads(l) for l in open(hf_hub_download(DATASET, BENCH, repo_type="dataset", token=tok,
                                                        local_dir=work, force_download=True), encoding="utf-8")
            if l.strip()]
    print(f"{len(rows)} next-edit rows ({sum(r['kind'] == 'edit' for r in rows)} with an edit)", flush=True)
    print("Doing nothing would score:", flush=True)
    nes_eval.show(nes_eval.score(rows, {r["id"]: nes_eval.shown_region(r) for r in rows}))

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

    def run_model(model):
        sha = version(model)
        name = f"{model}@{sha[:7]}"
        out = f"bench/nes_preds_{name.split('/')[-1]}.json"
        if not REDO and exists(out):
            print(f"\n== {name}: predictions already uploaded; their score:", flush=True)
            preds = json.load(open(os.path.join(work, out), encoding="utf-8"))["preds"]
            nes_eval.show(nes_eval.score(rows, preds))
            return
        print(f"\n== {name}", flush=True)
        subprocess.run("pkill -f 'vllm serve'", shell=True)
        time.sleep(5)
        log_path = os.path.join(work, "vllm.log")
        server = subprocess.Popen([f"{venv}/bin/vllm", "serve", model, "--revision", sha, "--tokenizer-revision", sha,
                                   "--port", "8000", "--max-model-len", "8192", "--gpu-memory-utilization", "0.85",
                                   "--max-num-seqs", "64"],
                                  env=dict(os.environ, HF_TOKEN=tok, VLLM_USE_FLASHINFER_SAMPLER="0"),
                                  stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        try:
            t0 = time.time()
            while True:
                if server.poll() is not None:
                    raise RuntimeError("vLLM exited while loading:\n" + open(log_path).read()[-3000:])
                try:
                    urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
                    break
                except Exception:
                    if time.time() - t0 > 30 * 60:
                        raise RuntimeError("vLLM did not start in 30 min:\n" + open(log_path).read()[-3000:])
                    time.sleep(10)
            print(f"vLLM serving {model} after {time.time() - t0:.0f}s", flush=True)
            url, done = "http://localhost:8000/v1/completions", [0]

            def one(r):
                try:
                    x = nes_eval.query_nes(url, model, r["prompt"])
                except Exception as e:
                    print("request failed:", str(e)[:200], flush=True)
                    x = ""
                done[0] += 1
                if done[0] % 500 == 0:
                    print(f"{done[0]}/{len(rows)}", flush=True)
                return r["id"], x

            t1 = time.time()
            with ThreadPoolExecutor(32) as pool:
                preds = dict(pool.map(one, rows))
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        empty = sum(not v.strip() for v in preds.values())
        print(f"{len(preds)} rewrites in {time.time() - t1:.0f}s ({empty} empty)", flush=True)
        assert empty < 0.5 * len(preds), "Most rewrites are empty: something went wrong, not uploading them"
        nes_eval.show(nes_eval.score(rows, preds))
        api.upload_file(path_or_fileobj=json.dumps({"key": {"data": BENCH, "n": len(rows)}, "preds": preds}).encode(),
                        path_in_repo=out, repo_id=DATASET, repo_type="dataset",
                        commit_message=f"next-edit predictions (vLLM): {name}")
        print(f"Predictions uploaded -> {DATASET}/{out}", flush=True)

    for m in MODELS:
        try:
            run_model(m)
        except Exception as e:
            print(f"!! {m} failed: {str(e)[-1500:]}", flush=True)
        finally:
            time.sleep(5)
            shutil.rmtree(os.path.join(HF_HUB_CACHE, "models--" + m.replace("/", "--")), ignore_errors=True)
    print("\nDone.", flush=True)
    return


if __name__ == "__main__":
    app.run()
