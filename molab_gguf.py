import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Ship a Quark model as GGUF (llama.cpp / Ollama / LM Studio), on molab's GPU notebook:
    #   1. download the model, convert it with llama.cpp (BF16), quantize to Q4_K_M and Q8_0
    #   2. upload both to <model>-GGUF on Hugging Face
    #   3. quick check: Q4_K_M writes fills for the 600 benchmark holes (llama-server on the GPU), uploaded to quark-data
    #      so GitHub > Actions > "Quark benchmark tests" (model = the -GGUF repo) can score them
    # Paste the HF token, turn the GPU on (notebook specs button at the top), click Start.
    import marimo as mo

    token = mo.ui.text(kind="password", label="Hugging Face token (hf_...)", full_width=True)
    start = mo.ui.run_button(label="Start GGUF")
    mo.vstack([token, start])
    return mo, start, token


@app.cell
def _(mo, start, token):
    import glob, json, os, shutil, signal, subprocess, sys, tarfile, time, urllib.request, zipfile
    from concurrent.futures import ThreadPoolExecutor

    mo.stop(not start.value, mo.md("Paste your Hugging Face token above, then click **Start GGUF**."))
    tok = token.value.strip()
    assert tok.startswith("hf_"), "Paste your Hugging Face token first"

    MODEL = "neutronai-code/Quark-12B-A2.5B-v0.7"
    GGUF_REPO = MODEL + "-GGUF"
    QUANTS = ["Q4_K_M", "Q8_0"]  # the first one gets the benchmark check
    LLAMA = "b12606"  # llama.cpp release (supports Mellum2's "mellum" architecture)
    DATASET = "neutronai-code/quark-data"
    BENCH = "bench/repo_holes_ctx2.jsonl"

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.hf_api import RepoFile

    api = HfApi(token=tok)

    def version(repo):  # as in ci_bench.py: the newest commit that changed the repo's own (non-.md) top-level files
        files = [f for f in api.list_repo_tree(repo, expand=True)
                 if isinstance(f, RepoFile) and f.last_commit and f.path != ".gitattributes"
                 and not f.path.endswith(".md")]
        sha = max(files, key=lambda f: f.last_commit.date).last_commit.oid if files else api.model_info(repo).sha
        return sha

    SHA = version(MODEL)
    base = MODEL.split("/")[-1]
    print(f"Converting {MODEL}@{SHA[:7]} -> {GGUF_REPO}", flush=True)

    def sh(cmd, cwd=None, env=None):  # run a command; show its output only if it fails
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, env=env)
        if r.returncode:
            raise RuntimeError(f"failed: {cmd}\n{(r.stdout + r.stderr)[-3000:]}")
        return r.stdout

    # The disk with the most room: the download (24 GB) and the BF16 GGUF (24 GB) sit there together for a while
    work = max([os.path.abspath("quark_work"), "/tmp/quark_gguf"],
               key=lambda d: shutil.disk_usage(os.path.dirname(d) or "/").free)
    os.makedirs(work, exist_ok=True)
    free = shutil.disk_usage(work).free / 1e9
    print(f"Working in {work} ({free:.0f} GB free)", flush=True)

    # NeutronAI's code (fim.py: the FIM prompt format; repo_fim.py: same_code), fresh every run
    code = os.path.join(work, "neutron-ai-main")
    shutil.rmtree(code, ignore_errors=True)
    zpath = os.path.join(work, "code.zip")
    urllib.request.urlretrieve("https://github.com/neutronai-code/neutron-ai/archive/refs/heads/main.zip", zpath)
    zipfile.ZipFile(zpath).extractall(work)
    sys.path.insert(0, code)
    import fim

    # llama.cpp: the source (for the converter) and the prebuilt CUDA binaries (llama-quantize, llama-server)
    llama = os.path.join(work, "llama.cpp")
    if not os.path.exists(os.path.join(llama, "src", "convert_hf_to_gguf.py")):
        shutil.rmtree(llama, ignore_errors=True)
        os.makedirs(os.path.join(llama, "bin"))
        rel = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA}"
        for url, to in [(f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{LLAMA}.tar.gz", "src"),
                        (f"{rel}/llama-{LLAMA}-bin-ubuntu-cuda-12.8-x64.tar.gz", "bin"),
                        (f"{rel}/cudart-llama-{LLAMA}-bin-ubuntu-cuda-12.8-x64.tar.gz", "bin")]:
            t = os.path.join(work, "dl.tar.gz")
            urllib.request.urlretrieve(url, t)
            tarfile.open(t).extractall(os.path.join(llama, to + "_x"), filter="tar")
            os.remove(t)
        os.rename(glob.glob(os.path.join(llama, "src_x", "*"))[0], os.path.join(llama, "src"))
    find = lambda name: next(p for p in glob.glob(os.path.join(llama, "bin_x", "**", name), recursive=True)
                             if os.path.isfile(p))
    quantize, server = find("llama-quantize"), find("llama-server")
    libdirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(llama, "bin_x", "**", "*.so*"),
                                                            recursive=True)})
    lenv = dict(os.environ, LD_LIBRARY_PATH=":".join(libdirs + [os.environ.get("LD_LIBRARY_PATH", "")]))
    for b in (quantize, server):
        os.chmod(b, 0o755)
    print("llama.cpp", LLAMA, "ready", flush=True)

    venv = "/tmp/gguf312"
    py = sys.executable
    if not os.path.exists(f"{venv}/bin/python"):
        sh(f"{py} -m pip install -q uv")
        sh(f"{py} -m uv venv {venv} --python 3.12 --clear")
        sh(f"{py} -m uv pip install --python {venv}/bin/python -r requirements/requirements-convert_hf_to_gguf.txt",
           cwd=os.path.join(llama, "src"))

    out = os.path.join(work, "gguf")
    os.makedirs(out, exist_ok=True)
    bf16 = os.path.join(out, f"{base}-BF16.gguf")
    files = {q: os.path.join(out, f"{base}-{q}.gguf") for q in QUANTS}
    # GGUF files already on Hugging Face from an earlier run (name -> size): those aren't made or uploaded again
    online = ({f.path: f.size for f in api.list_repo_tree(GGUF_REPO) if isinstance(f, RepoFile)}
              if api.repo_exists(GGUF_REPO) else {})
    done_q = [q for q, f in files.items() if os.path.basename(f) in online]
    if done_q:
        print(f"Already on Hugging Face: {', '.join(done_q)}", flush=True)
    if len(done_q) < len(QUANTS) and not all(os.path.exists(f) for f in files.values()):
        assert free > 55, f"Only {free:.0f} GB free in {work}: restart the notebook for a clean disk, then run again"
        t0 = time.time()
        hf = os.path.join(work, "hf")
        snapshot_download(MODEL, revision=SHA, local_dir=hf, token=tok, ignore_patterns=["last-checkpoint/*", "*.md"])
        print(f"Downloaded in {(time.time() - t0) / 60:.0f} min", flush=True)
        # The fine-tune's config/tokenizer files were saved by transformers 5 ("num_local_experts",
        # "TokenizersBackend"), which llama.cpp's converter (transformers 4.57) doesn't read. Training changed neither
        # (same architecture; vocab, merges and special tokens checked identical), so use the base model's copies.
        for f in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json"]:
            shutil.copyfile(hf_hub_download("JetBrains/Mellum2-12B-A2.5B-Base", f, token=tok), os.path.join(hf, f))
        t0 = time.time()
        sh(f"{venv}/bin/python convert_hf_to_gguf.py {hf} --outtype bf16 --outfile {bf16}", cwd=os.path.join(llama, "src"))
        shutil.rmtree(hf)
        print(f"Converted to BF16 GGUF in {(time.time() - t0) / 60:.0f} min ({os.path.getsize(bf16) / 1e9:.1f} GB)",
              flush=True)
        for q, f in files.items():
            t0 = time.time()
            sh(f"{quantize} {bf16} {f} {q} {os.cpu_count()}", env=lenv)
            print(f"{q}: {os.path.getsize(f) / 1e9:.1f} GB in {(time.time() - t0) / 60:.0f} min", flush=True)
        os.remove(bf16)

    api.create_repo(GGUF_REPO, exist_ok=True)
    for q, f in files.items():
        if q in done_q:
            continue
        t0 = time.time()
        api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=GGUF_REPO,
                        commit_message=f"{q} GGUF of {MODEL}@{SHA[:7]} (llama.cpp {LLAMA})")
        print(f"Uploaded {os.path.basename(f)} in {(time.time() - t0) / 60:.0f} min", flush=True)
    print(f"GGUF files live -> https://huggingface.co/{GGUF_REPO}", flush=True)

    # Quick check: does the quantized model still complete code like the full one? Fills for the benchmark holes
    NAME = f"{GGUF_REPO}@{version(GGUF_REPO)[:7]}"
    q = QUANTS[0]
    if not os.path.exists(files[q]):  # a fresh notebook: get it back from Hugging Face
        print(f"Downloading {os.path.basename(files[q])} for the check", flush=True)
        files[q] = hf_hub_download(GGUF_REPO, os.path.basename(files[q]), token=tok, local_dir=out)
    rows =[json.loads(l) for l in open(hf_hub_download(DATASET, BENCH, repo_type="dataset", token=tok,
                                                        local_dir=work, force_download=True), encoding="utf-8")
            if l.strip()]
    fim._FIM_STYLE[NAME] = "starcoder"
    log_path = os.path.join(work, "server.log")
    srv = subprocess.Popen([server, "-m", files[q], "-ngl", "999", "-c", str(8 * 8192), "-np", "8", "--port", "8080"],
                           env=lenv, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        # Wait until the model really answers: this llama-server says "healthy" while it is still loading the model
        # (completions then fail with 503 "Loading model"), so send a tiny completion until one works
        t0 = time.time()
        url, done = "http://localhost:8080/v1/completions", [0]
        probe = json.dumps({"model": NAME, "prompt": "def", "max_tokens": 1}).encode()
        while True:
            if srv.poll() is not None:
                raise RuntimeError("llama-server exited; the end of its log:\n" + open(log_path).read()[-3000:])
            try:
                urllib.request.urlopen(urllib.request.Request(url, probe, {"Content-Type": "application/json"}),
                                       timeout=60)
                break
            except Exception:
                if time.time() - t0 > 20 * 60:
                    raise RuntimeError("llama-server did not load the model in 20 min; the end of its log:\n"
                                       + open(log_path).read()[-3000:])
                time.sleep(10)
        print(f"llama-server ready after {(time.time() - t0) / 60:.1f} min", flush=True)

        def one(p):
            x = ""
            for attempt in range(3):
                try:
                    x = fim.extract_fim_middle(fim.query_fim_samples(url, NAME, p, max_tokens=512)[0])
                    break
                except Exception as e:
                    print("request failed:", str(e)[:200], flush=True)
                    time.sleep(10)
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"{done[0]}/{len(rows)} holes", flush=True)
            return x

        t1 = time.time()
        with ThreadPoolExecutor(8) as pool:
            fills = list(pool.map(one, [r["fim_prompt"] for r in rows]))
        print(f"{q} wrote {len(fills)} fills in {(time.time() - t1) / 60:.0f} min", flush=True)
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    exact = sum(f.strip() == r["oracle_middle"].strip() for f, r in zip(fills, rows))
    msg = f"{q}: {exact} fills match the original code exactly"
    try:  # compare with the full (bf16) model's fills on the same holes
        import repo_fim

        full = f"{MODEL}@{SHA[:7]}"
        ref = json.load(open(hf_hub_download(DATASET, f"bench/fills_{full.split('/')[-1]}_ctx2.json",
                                             repo_type="dataset", token=tok, local_dir=work), encoding="utf-8"))
        ref = ref["fills"][full]
        same = sum(repo_fim.same_code(a, b) for a, b in zip(fills, ref))
        msg += (f" (full model: {sum(f.strip() == r['oracle_middle'].strip() for f, r in zip(ref, rows))}); "
                f"{same}/{len(rows)} fills identical to the full model's")
    except Exception as e:
        msg += f" (no comparison with the full model: {e})"
    print(msg, flush=True)
    empty = sum(not f.strip() for f in fills)
    assert empty < 0.1 * len(fills), (f"{empty}/{len(fills)} fills are empty, so something went wrong: not uploading "
                                      f"them. The end of llama-server's log:\n" + open(log_path).read()[-3000:])
    api.upload_file(
        path_or_fileobj=json.dumps({"key": {"data": "repo_holes_ctx2.jsonl", "ids": [r["id"] for r in rows]},
                                    "fills": {NAME: fills}}).encode("utf-8"),
        path_in_repo=f"bench/fills_{NAME.split('/')[-1]}_ctx2.json", repo_id=DATASET, repo_type="dataset",
        commit_message=f"benchmark fills: {NAME} ({q}, llama.cpp {LLAMA})")
    print(f"Fills uploaded. Next: GitHub > Actions > Quark benchmark tests > Run workflow, model = {GGUF_REPO}",
          flush=True)
    return


if __name__ == "__main__":
    app.run()
