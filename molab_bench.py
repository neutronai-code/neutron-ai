import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # The whole Quark benchmark on molab (free), no Modal needed:
    #   1. fills (GPU): vLLM 0.33 (the Modal benchmark's version) writes MODEL's greedy fill for each of the 600 holes
    #      and uploads them to the private quark-data dataset. Skipped when that model's fills are already there,
    #      so this notebook also runs WITHOUT a GPU (e.g. next to a training notebook) once the fills exist.
    #   2. tests (CPU): each fill is spliced into its real repo and that repo's own tests decide pass/fail
    #      (local_repo_eval.py: same scoring as Modal). The table compares MODEL with the earlier models, and the
    #      per-hole report goes back to quark-data/bench/report_<CTX>.jsonl.
    # Paste the HF token, click Start. For fills, turn the GPU on first (notebook specs button at the top).
    import marimo as mo

    token = mo.ui.text(kind="password", label="Hugging Face token (hf_...)", full_width=True)
    start = mo.ui.run_button(label="Start benchmark")
    mo.vstack([token, start])
    return mo, start, token


@app.cell
def _(mo, start, token):
    import json, os, shutil, signal, subprocess, sys, time, urllib.request, zipfile
    from concurrent.futures import ThreadPoolExecutor

    mo.stop(not start.value, mo.md("Paste your Hugging Face token above, then click **Start benchmark**."))
    tok = token.value.strip()
    assert tok.startswith("hf_"), "Paste your Hugging Face token first"

    MODEL = "neutronai-code/Quark-12B-A2.5B-v0.7"
    REDO_FILLS = False  # True: make new fills even if this model's are already uploaded (e.g. the model changed)
    LIMIT = 0  # smoke test: test only the first N holes of each repo (0: all 600)
    WORKERS = 4  # test batches at once: molab gives 4 CPUs
    # This notebook's memory in GB (the specs button at the top: 4 CPU · 32 GiB, GPU or not): the tests wait before
    # starting a batch above 75% of this.
    MEM_GB = 32
    DATASET = "neutronai-code/quark-data"
    # Which prompts: "ctx2" = the 600 holes with editor-style (names) context, 4k chars (the main benchmark);
    # "ctx3" = the same with 8k chars. Each has its own fills and report; test results are shared (same hole + same
    # fill = same result, whatever the prompt was).
    CTX = "ctx2"
    BENCH = f"bench/repo_holes_{CTX}.jsonl"
    REPORT = f"bench/report_{CTX}.jsonl"  # every model's per-hole results, so the table shows them side by side
    work = os.path.abspath("quark_work")
    os.makedirs(work, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download

    # Results are kept per model VERSION, so e.g. v0.7 halfway through training and the finished v0.7 get their own
    # fills and their own column in the table. The version is the newest commit that changed the model's own files
    # (top folder), not the repo's latest commit: a training run commits its last-checkpoint/ every 100 steps
    # without changing the model, and that must neither look like a newer model nor redo its fills.
    from huggingface_hub.hf_api import RepoFile
    _files = [f for f in HfApi(token=tok).list_repo_tree(MODEL, expand=True)
              if isinstance(f, RepoFile) and f.last_commit and f.path != ".gitattributes"
              and not f.path.endswith(".md")]
    SHA = (max(_files, key=lambda f: f.last_commit.date).last_commit.oid if _files
           else HfApi(token=tok).model_info(MODEL).sha)
    NAME = f"{MODEL}@{SHA[:7]}"
    OUT = f"bench/fills_{NAME.split('/')[-1]}_{CTX}.json"
    print("Benchmarking", NAME, flush=True)

    def sh(cmd):  # run a command; show its output only if it fails (marimo hides subprocess output)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"failed: {cmd}\n{(r.stdout + r.stderr)[-3000:]}")
        return r.stdout

    def get(name):  # a file from the quark-data dataset (None if it isn't there)
        try:
            return hf_hub_download(DATASET, name, repo_type="dataset", token=tok, local_dir=work, force_download=True)
        except Exception as e:
            if "404" in str(e) or "EntryNotFound" in type(e).__name__:
                return None
            raise

    # 1. NeutronAI's code, fresh every run so fixes arrive (fim.py: the FIM prompt format; local_repo_eval.py: tests)
    code = os.path.join(work, "neutron-ai-main")
    shutil.rmtree(code, ignore_errors=True)
    zpath = os.path.join(work, "code.zip")
    urllib.request.urlretrieve("https://github.com/neutronai-code/neutron-ai/archive/refs/heads/main.zip", zpath)
    zipfile.ZipFile(zpath).extractall(work)
    sys.path.insert(0, code)
    import fim

    rows = [json.loads(l) for l in open(get(BENCH), encoding="utf-8") if l.strip()]
    print(f"{len(rows)} benchmark holes", flush=True)

    # 2. Fills: reuse the uploaded ones, else make them on the GPU
    old = None if REDO_FILLS else get(OUT)
    if old:
        fills = json.load(open(old, encoding="utf-8"))["fills"][NAME]
        print(f"Using the fills already uploaded for {NAME} ({OUT}): no GPU needed", flush=True)
    else:
        # The FIM format comes from the model's tokenizer.json: a fine-tune saved by transformers 5 has no
        # special-token list in tokenizer_config.json, so Mellum fine-tunes must not be guessed from that.
        with open(hf_hub_download(MODEL, "tokenizer.json", revision=SHA, token=tok), encoding="utf-8") as _f:
            _t = _f.read()
        fim._FIM_STYLE[MODEL] = "starcoder" if '"<fim_prefix>"' in _t else "granite" if '"<|filename|>"' in _t else "qwen"
        print("FIM prompt format:", fim._FIM_STYLE[MODEL], "(Mellum must be starcoder)", flush=True)

        # vLLM 0.33 on Python 3.12 in its own environment (molab's Python 3.13 can't run it). vLLM 0.33 has no
        # vllm._C module any more: the check is whether it detects the GPU.
        gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                             capture_output=True, text=True)
        assert gpu.returncode == 0 and gpu.stdout.strip(), (
            f"No GPU, and no fills uploaded yet for {MODEL}: open the notebook specs button at the top, pick the "
            "RTX Pro 6000 GPU, Apply and restart, then run again")
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

        # Serve the model (same settings as the Modal benchmark). vLLM wants 85% of the GPU, so first stop a vLLM
        # left over from an earlier run; a model loaded in this notebook's own memory needs a notebook restart.
        subprocess.run("pkill -f 'vllm serve'", shell=True)
        time.sleep(5)
        free = sh("nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits").split(",")
        print(f"GPU memory free: {int(free[0]) / 1024:.0f} of {int(free[1]) / 1024:.0f} GiB", flush=True)
        assert int(free[0]) > 0.86 * int(free[1]), ("The GPU is still in use (e.g. a model from an earlier run in "
                                                    "this notebook): restart the notebook, then run it again")
        log_path = os.path.join(work, "vllm.log")
        server = subprocess.Popen([f"{venv}/bin/vllm", "serve", MODEL, "--revision", SHA, "--tokenizer-revision", SHA,
                                   "--port", "8000", "--max-model-len", "8192",
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
                        raise RuntimeError("vLLM did not start in 30 min; the end of its log:\n"
                                           + open(log_path).read()[-3000:])
                    time.sleep(10)
            print(f"vLLM serving {MODEL} after {time.time() - t0:.0f}s", flush=True)

            # Greedy fills, exactly like modal_repo_fim.generate
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

            print("sample fill:\n" + one(rows[0]["fim_prompt"])[:300] + "\n--- original:\n"
                  + rows[0]["oracle_middle"][:300], flush=True)
            done[0] = 0
            t1 = time.time()
            with ThreadPoolExecutor(32) as pool:
                fills = list(pool.map(one, [r["fim_prompt"] for r in rows]))
            exact = sum(f.strip() == r["oracle_middle"].strip() for f, r in zip(fills, rows))
            med = sorted(len(f) for f in fills)[len(fills) // 2]
            print(f"{len(fills)} fills in {time.time() - t1:.0f}s: {exact} match the original exactly (untrained "
                  f"Mellum2: 165); median {med} chars (original code: ~98); {sum(not f for f in fills)} empty",
                  flush=True)
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:  # already exited
                pass
        HfApi(token=tok).upload_file(
            path_or_fileobj=json.dumps({"key": {"data": f"repo_holes_{CTX}.jsonl", "ids": [r["id"] for r in rows]},
                                        "fills": {NAME: fills}}).encode("utf-8"),
            path_in_repo=OUT, repo_id=DATASET, repo_type="dataset", commit_message=f"benchmark fills (vLLM): {NAME}")
        print(f"Fills uploaded -> {DATASET}/{OUT}", flush=True)

    # 3. Tests on this machine's CPUs (installs Node 24, GCC 14, CMake, uv the first time: a few minutes)
    fills_path = os.path.join(work, "fills_now.json")
    with open(fills_path, "w", encoding="utf-8") as _f:
        json.dump({"key": {"data": f"repo_holes_{CTX}.jsonl", "ids": [r["id"] for r in rows]}, "fills": {NAME: fills}}, _f)
    report = get(REPORT) or os.path.join(work, REPORT)
    info = get("bench/quark_repos.jsonl")  # the repos' URLs and pinned commits
    test_log = os.path.join(work, "tests.log")
    proc = subprocess.Popen([sys.executable, "-u", "local_repo_eval.py", "--data", os.path.join(work, BENCH),
                             "--fills", fills_path, "--report", report, "--repo-info", info,
                             "--workers", str(WORKERS), "--limit", str(LIMIT), "--show", "4",
                             "--mem-gb", str(MEM_GB),
                             # finished batches, also kept on Hugging Face (a restart wipes this machine's disk):
                             # rerunning continues instead of starting over
                             "--progress", os.path.join(work, f"progress_{NAME.split('/')[-1]}.jsonl"),
                             "--hub-progress", f"{DATASET}:bench/progress_{NAME.split('/')[-1]}.jsonl"],
                            cwd=code, env=dict(os.environ, HF_TOKEN=tok),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    with open(test_log, "w") as _log:
        for line in proc.stdout:
            _log.write(line)
            print(line.rstrip(), flush=True)
    assert proc.wait() == 0, f"the tests crashed (full log: {test_log})"

    # 4. Keep the per-hole results (full runs only, so a smoke test can't replace real scores)
    if not LIMIT:
        HfApi(token=tok).upload_file(path_or_fileobj=report, path_in_repo=REPORT, repo_id=DATASET,
                                     repo_type="dataset", commit_message=f"benchmark results: {NAME}")
        print(f"\nResults uploaded -> {DATASET}/{REPORT}. Done!")
    return


if __name__ == "__main__":
    app.run()
