import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Benchmark fills on molab's GPU for one or more models, one after another: vLLM 0.33 writes each model's greedy
    # fill for the 600 benchmark holes of each prompt set it's listed with, in that model's own FIM format
    # (fim.fim_style), and uploads them to the private quark-data dataset. Fills already there are skipped.
    # Then test them: GitHub > Actions > "Quark benchmark tests" > Run workflow, one run per prompt set, with the
    # models comma-separated in the model box (the printout at the end gives the exact text).
    # Paste the HF token, turn the GPU on (notebook specs button at the top), click Start.
    import marimo as mo

    token = mo.ui.text(kind="password", label="Hugging Face token (hf_...)", full_width=True)
    start = mo.ui.run_button(label="Start fills")
    mo.vstack([token, start])
    return mo, start, token


@app.cell
def _(mo, start, token):
    import json, os, shutil, signal, subprocess, sys, time, urllib.request, zipfile
    from concurrent.futures import ThreadPoolExecutor

    mo.stop(not start.value, mo.md("Paste your Hugging Face token above, then click **Start fills**."))
    tok = token.value.strip()
    assert tok.startswith("hf_"), "Paste your Hugging Face token first"

    # (model, prompt sets). Prompt sets (bench/repo_holes_<set>.jsonl, make_ctx_bench.py): "ctx2" = the main
    # benchmark (4k context); "ctx3" = 8k; "light" = the IDE's Light profile for slow machines (2k context, 1,600 /
    # 600 chars around the hole); "fastlight" = Light with the short suffix the IDE gives Quark-1.5B (<=300 chars).
    JOBS = [
        ("neutronai-code/Quark-12B-A2.5B-v0.8-nes", ["ctx2"]),  # next-edit training: must stay at v0.7's 59.8%
        ("neutronai-code/Quark-12B-A2.5B-v0.7", ["light"]),  # what the Light profile costs (v0.7 ctx2: 59.8%)
        ("neutronai-code/Quark-1.5B-v0.5", ["light", "fastlight"]),  # the IDE's Fast model (ctx2: 52.2%)
    ]
    # Extra vLLM settings some models need. Codestral: read its Hugging Face files, not Mistral's own format (whose
    # tokenizer would treat "[SUFFIX]" in a raw prompt as plain text instead of the FIM token).
    EXTRA = {"mistralai/Codestral-22B-v0.1": ["--tokenizer-mode", "hf", "--config-format", "hf",
                                              "--load-format", "safetensors"]}
    REDO_FILLS = False  # True: make new fills even if a model's are already uploaded
    DATASET = "neutronai-code/quark-data"
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

    def sh(cmd):  # run a command; show its output only if it fails (marimo hides subprocess output)
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

    # NeutronAI's code (fim.py: every model's FIM prompt format), fresh every run
    code = os.path.join(work, "neutron-ai-main")
    shutil.rmtree(code, ignore_errors=True)
    zpath = os.path.join(work, "code.zip")
    urllib.request.urlretrieve("https://github.com/neutronai-code/neutron-ai/archive/refs/heads/main.zip", zpath)
    zipfile.ZipFile(zpath).extractall(work)
    sys.path.insert(0, code)
    import fim

    holes = {}
    for ctx in sorted({c for _, cs in JOBS for c in cs}):
        holes[ctx] = [json.loads(l) for l in open(hf_hub_download(DATASET, f"bench/repo_holes_{ctx}.jsonl",
                                                                  repo_type="dataset", token=tok, local_dir=work,
                                                                  force_download=True), encoding="utf-8")
                      if l.strip()]
        print(f"{len(holes[ctx])} benchmark holes ({ctx})", flush=True)

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

    def make_fills(model, ctxs):
        sha = version(model)
        name = f"{model}@{sha[:7]}"
        outs = {ctx: f"bench/fills_{name.split('/')[-1]}_{ctx}.json" for ctx in ctxs}
        todo = [ctx for ctx in ctxs if REDO_FILLS or not exists(outs[ctx])]
        for ctx in ctxs:
            if ctx not in todo:
                print(f"\n== {name} ({ctx}): fills already uploaded, skipped", flush=True)
        if not todo:
            return
        style = fim.fim_style(model)
        print(f"\n== {name}: FIM format {style}", flush=True)
        subprocess.run("pkill -f 'vllm serve'", shell=True)
        time.sleep(5)
        free = sh("nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits").split(",")
        assert int(free[0]) > 0.86 * int(free[1]), "The GPU is still in use: restart the notebook, then run it again"
        log_path = os.path.join(work, "vllm.log")
        server = subprocess.Popen([f"{venv}/bin/vllm", "serve", model, "--revision", sha, "--tokenizer-revision", sha,
                                   "--port", "8000", "--max-model-len", "8192", "--gpu-memory-utilization", "0.85",
                                   "--max-num-seqs", "64", *EXTRA.get(model, [])],
                                  env=dict(os.environ, HF_TOKEN=tok, VLLM_USE_FLASHINFER_SAMPLER="0"),
                                  stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        try:
            t0 = time.time()
            while True:
                if server.poll() is not None:
                    raise RuntimeError("vLLM exited while loading; the end of its log:\n"
                                       + open(log_path).read()[-3000:])
                try:
                    urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
                    break
                except Exception:
                    if time.time() - t0 > 30 * 60:
                        raise RuntimeError("vLLM did not start in 30 min:\n" + open(log_path).read()[-3000:])
                    time.sleep(10)
            print(f"vLLM serving {model} after {time.time() - t0:.0f}s", flush=True)
            url, done = "http://localhost:8000/v1/completions", [0]

            def one(p):
                try:
                    x = fim.extract_fim_middle(fim.query_fim_samples(url, model, p, max_tokens=512)[0])
                except Exception as e:
                    print("request failed:", e, flush=True)
                    x = ""
                done[0] += 1
                if done[0] % 200 == 0:
                    print(f"{done[0]}/{len(rows)} fills", flush=True)
                return x

            for ctx in todo:
                rows = holes[ctx]
                print(f"-- {ctx}: sample fill:\n" + one(rows[0]["fim_prompt"])[:300] + "\n--- original:\n"
                      + rows[0]["oracle_middle"][:300], flush=True)
                done[0] = 0
                t1 = time.time()
                with ThreadPoolExecutor(32) as pool:
                    fills = list(pool.map(one, [r["fim_prompt"] for r in rows]))
                exact = sum(f.strip() == r["oracle_middle"].strip() for f, r in zip(fills, rows))
                med = sorted(len(f) for f in fills)[len(fills) // 2]
                print(f"{ctx}: {len(fills)} fills in {time.time() - t1:.0f}s: {exact} match the original exactly "
                      f"(ctx2: Quark v0.7 183, Mellum2 base 165); median {med} chars (original code: ~98); "
                      f"{sum(not f for f in fills)} empty", flush=True)
                assert sum(not f for f in fills) < 0.5 * len(fills), "Most fills are empty: not uploading them"
                api.upload_file(
                    path_or_fileobj=json.dumps({"key": {"data": f"repo_holes_{ctx}.jsonl",
                                                        "ids": [r["id"] for r in rows]},
                                                "fills": {name: fills}}).encode("utf-8"),
                    path_in_repo=outs[ctx], repo_id=DATASET, repo_type="dataset",
                    commit_message=f"benchmark fills (vLLM, {ctx}): {name}")
                print(f"Fills uploaded -> {DATASET}/{outs[ctx]}", flush=True)
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    ready, failed = {}, []
    for m, ctxs in JOBS:
        try:
            make_fills(m, ctxs)
            for ctx in ctxs:
                ready.setdefault(ctx, []).append(m)
        except Exception as e:  # one model failing (e.g. a license not accepted) shouldn't stop the others
            print(f"!! {m} failed: {str(e)[-1500:]}", flush=True)
            failed.append(m)
        finally:  # the next model needs the disk: each download is 10-45 GB
            time.sleep(5)
            shutil.rmtree(os.path.join(HF_HUB_CACHE, "models--" + m.replace("/", "--")), ignore_errors=True)
    print("\nDone." + (f" Failed: {', '.join(failed)}" if failed else ""), flush=True)
    if ready:
        print("Next: GitHub > Actions > Quark benchmark tests > Run workflow, once per line (model box, ctx):",
              flush=True)
        for ctx, ms in ready.items():
            print(f"  {','.join(ms)}   ctx: {ctx}", flush=True)
    return


if __name__ == "__main__":
    app.run()
