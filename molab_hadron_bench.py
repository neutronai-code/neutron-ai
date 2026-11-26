import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Hadron answers on molab's GPU, step 1 of 2: vLLM 0.33 answers every task 4 times (temperature 0.6, thinking on,
    # 32k tokens per request) for each model you tick, one after another, and uploads the answers to the hadron-data
    # dataset (gen/<model>-32k/). Tasks: the bake-off set (236 held-out + 26 hand-made: the benchmark), or the 2,871
    # training tasks (pass rates: which tasks go to RL and which to teacher distillation).
    # Step 2, the tests, runs off molab (hadron_score.py): GPU notebooks that also ran tests kept being reclaimed,
    # so this one only generates, like the Quark fills notebook.
    # Crash-safe: answers are uploaded every few minutes, and a rerun continues where the last one stopped.
    # Progress (and the machine's state) also goes to gen/<model>-32k/progress.txt, since molab's output often
    # stops refreshing. A Hadron checkpoint is only a LoRA adapter: it is merged into Qwen3.5-4B here.
    # Paste the HF token, tick the models, turn the GPU on (notebook specs button at the top), click Start.
    # Run ONE notebook at a time.
    import marimo as mo

    # name -> (repo, Hadron checkpoint step or 0 for a plain model, GPU memory it needs in GB, extra vLLM flags)
    MODELS = {
        "Qwen3.5-4B (today's Hadron base: the baseline)": ("Qwen/Qwen3.5-4B", 0, 20, []),
        "Mellum2.1-12B-A2.5B-Thinking (JetBrains)": ("JetBrains/Mellum2.1-12B-A2.5B-Thinking", 0, 36, []),
        "Hadron Mellum2.1 SFT pilot (merged)": ("neutronai-code/Hadron-Mellum2.1-12B-A2.5B-sft1", 0, 36, []),
        "Qwen3.5-9B": ("Qwen/Qwen3.5-9B", 0, 32, []),
        "Granite 4.2 8B (IBM)": ("ibm-granite/granite-4.2-8b", 0, 40, []),
        "Qwen3.6-35B-A3B (big)": ("Qwen/Qwen3.6-35B-A3B", 0, 84, ["--language-model-only"]),
        "Hadron-4B-p1 step 400": ("neutronai-code/Hadron-4B-p1", 400, 20, []),
        "Hadron-4B-p1 step 200": ("neutronai-code/Hadron-4B-p1", 200, 20, []),
    }
    TASK_SETS = {
        "Bake-off: 236 held-out + 26 hand-made tasks": "bench",
        "Training tasks: all 2,871 others (pass rates for RL and distillation; ~4-6 h per model)": "train",
    }
    token = mo.ui.text(kind="password", label="Hugging Face token (hf_...)", full_width=True)
    choice = mo.ui.multiselect(options=list(MODELS), value=[list(MODELS)[1]],
                               label="Models to run (in this order)", full_width=True)
    taskset = mo.ui.dropdown(options=list(TASK_SETS), value=list(TASK_SETS)[1], label="Tasks", full_width=True)
    start = mo.ui.run_button(label="Start Hadron answers")
    mo.vstack([token, choice, taskset, start])
    return MODELS, TASK_SETS, choice, mo, start, taskset, token


@app.cell
def _(MODELS, TASK_SETS, choice, mo, start, taskset, token):
    import json, os, shutil, signal, socket, subprocess, sys, textwrap, threading, time, urllib.error, urllib.request, zipfile
    from concurrent.futures import ThreadPoolExecutor

    mo.stop(not start.value, mo.md("Paste your Hugging Face token, tick the models, then click **Start Hadron answers**."))
    tok = token.value.strip()
    assert tok.startswith("hf_"), "Paste your Hugging Face token first"
    assert choice.value, "Tick at least one model"

    BASE, BASE_REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # the base Hadron was trained on
    DATASET = "neutronai-code/hadron-data"
    CTX = 32768  # tokens per request (prompt + thinking + answer), the same for every model
    N, TEMP = 4, 0.6  # answers per task and their temperature, as in modal_eval.py
    TRAIN = TASK_SETS[taskset.value] == "train"
    PART = 250  # training tasks are answered and uploaded in parts of this many (a crash loses at most one part)
    CONC = 128 if TRAIN else 64  # requests at once (= vLLM's --max-num-seqs); results don't depend on it

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    api = HfApi(token=tok)
    work = os.path.abspath("hadron_work")  # like the fills notebook: the notebook's own folder, not /tmp
    os.makedirs(work, exist_ok=True)
    log_lines, stage = [], ["starting"]

    def log(msg):
        print(msg, flush=True)
        log_lines.append(f"{time.strftime('%H:%M:%S')} {msg}")

    def read(path):
        try:
            return open(path).read().strip()
        except OSError:
            return ""

    def machine():  # one status line: if molab kills the notebook, the last progress.txt on the Hub shows the state
        info = dict(l.split(":", 1) for l in read("/proc/meminfo").splitlines() if ":" in l)
        avail = int(info.get("MemAvailable", "0 kB").split()[0]) / 1e6
        procs = threads = 0
        for p in os.listdir("/proc"):
            if p.isdigit():
                procs += 1
                for l in read(f"/proc/{p}/status").splitlines():
                    if l.startswith("Threads:"):
                        threads += int(l.split()[1])
        mem = read("/sys/fs/cgroup/memory.current") or read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        return (f"load {os.getloadavg()[0]:.1f} on {os.cpu_count()} CPUs | RAM available {avail:.1f} GB, cgroup "
                f"{int(mem) / 1e9 if mem.isdigit() else 0:.1f} GB | {procs} processes, {threads} threads | "
                f"/tmp {shutil.disk_usage('/tmp').used / 1e9:.0f} GB, {work} {shutil.disk_usage(work).used / 1e9:.0f} GB")

    def sh(cmd, **kw):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
        if r.returncode:
            log(f"FAILED ({stage[0]}): {(r.stdout + r.stderr)[-1500:]}")
            raise RuntimeError(f"failed: {cmd}\n{(r.stdout + r.stderr)[-3000:]}")
        return r.stdout

    # 1. Stop what an earlier run of this notebook left behind (a restarted notebook keeps its old processes).
    pidfile = f"{work}/pids.txt"
    for pid in (read(pidfile).split() if os.path.exists(pidfile) else []):
        try:
            os.killpg(int(pid), signal.SIGTERM)
            log(f"stopped leftover process group {pid} from an earlier run")
        except (ProcessLookupError, PermissionError, ValueError):
            pass
    open(pidfile, "w").close()

    # 2. The GPU and the tasks.
    gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits",
                         shell=True, capture_output=True, text=True)
    assert gpu.returncode == 0 and gpu.stdout.strip(), (
        "No GPU: open the notebook specs button at the top, pick the RTX Pro 6000 GPU, Apply and restart, then run again")
    gname, gtotal, _gfree = [x.strip() for x in gpu.stdout.strip().splitlines()[0].split(",")]
    gtotal = float(gtotal) / 1024
    log(f"GPU: {gname} ({gtotal:.0f} GB) | {machine()}")
    code = f"{work}/neutron-ai-main"  # NeutronAI's code: the exact prompts eval.py sends (repair.py), and the tasks
    shutil.rmtree(code, ignore_errors=True)
    urllib.request.urlretrieve("https://github.com/neutronai-code/neutron-ai/archive/refs/heads/main.zip", f"{work}/code.zip")
    zipfile.ZipFile(f"{work}/code.zip").extractall(work)
    sys.path.insert(0, code)
    from repair import system_prompt_for, task_prompt

    def load(name):
        return [json.loads(l) for l in open(f"{code}/{name}", encoding="utf-8") if l.strip()]

    # Each "file" is one upload: gen/<TAG>/<file>.json. Bake-off: tasks_holdout + tasks. Training: every generated task
    # that isn't held out, in file order, as train/part-000, train/part-001, ... (hadron_score.py --set train reads them).
    if TRAIN:
        held_out = {t["id"] for t in load("tasks_holdout.jsonl")}
        pool_ = [t for t in load("tasks_generated.jsonl") if t["id"] not in held_out]
        tasks = {f"train/part-{i // PART:03d}": pool_[i:i + PART] for i in range(0, len(pool_), PART)}
        log(f"training tasks: {len(pool_)} in {len(tasks)} parts of up to {PART}")
    else:
        tasks = {f: load(f"{f}.jsonl") for f in ["tasks_holdout", "tasks"]}
        log(", ".join(f"{f}: {len(t)} tasks" for f, t in tasks.items()))
    FILES = list(tasks)

    # 3. vLLM 0.33 in a Python 3.12 venv: the same install as the fills notebook.
    venv = "/tmp/vllm312"
    py = sys.executable
    if not os.path.exists(f"{venv}/bin/vllm"):
        stage[0] = "installing vLLM 0.33"
        log("[vllm] installing vLLM 0.33 (5-10 minutes)")
        sh(f"{py} -m pip install -q uv")
        sh(f"{py} -m uv venv {venv} --python 3.12 --clear")
        sh(f"{py} -m uv pip install --python {venv}/bin/python vllm==0.33.0")
        log(f"[vllm] installed | {machine()}")
    if not os.path.exists(f"{venv}/.kernels"):  # optional speed-up: precompiled FlashInfer kernels for this GPU
        try:
            sh(f"{venv}/bin/vllm download-kernels")
            open(f"{venv}/.kernels", "w").close()
        except RuntimeError as e:
            print("(kernel download skipped:", str(e).strip().splitlines()[-1][:200] + ")", flush=True)

    MERGE = textwrap.dedent(r'''
        # Merge a LoRA adapter into its base by hand: W += alpha/r * B @ A per layer, base files and names kept.
        import json, os, re, shutil, sys, torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        adapter, base, out = sys.argv[1:4]
        cfg = json.load(open(f"{adapter}/adapter_config.json"))
        assert cfg["peft_type"] == "LORA" and not cfg.get("use_dora") and not cfg.get("fan_in_fan_out"), cfg
        assert not cfg.get("use_rslora") and not cfg.get("rank_pattern") and not cfg.get("alpha_pattern"), cfg
        scale = cfg["lora_alpha"] / cfg["r"]
        lora = load_file(f"{adapter}/adapter_model.safetensors")
        mods = {}
        for k, v in lora.items():
            m = re.fullmatch(r"base_model\.model\.(.+)\.(lora_[AB])\.weight", k)
            assert m, f"unexpected adapter key {k}"
            mods.setdefault(m.group(1) + ".weight", {})[m.group(2)] = v
        index = json.load(open(f"{base}/model.safetensors.index.json"))
        missing = [w for w in mods if w not in index["weight_map"]]
        assert not missing, f"adapter layers not in the base: {missing[:5]}"
        os.makedirs(out)
        done, changed = 0, 0.0
        for shard in sorted(set(index["weight_map"].values())):
            with safe_open(f"{base}/{shard}", "pt") as f:
                meta = f.metadata()
            t = load_file(f"{base}/{shard}")
            for w, ab in mods.items():
                if w in t:
                    W = t[w]
                    delta = scale * (ab["lora_B"].float() @ ab["lora_A"].float())
                    assert delta.shape == W.shape, (w, delta.shape, W.shape)
                    t[w] = (W.float() + delta).to(W.dtype)
                    changed = max(changed, (t[w].float() - W.float()).abs().max().item())
                    done += 1
            save_file(t, f"{out}/{shard}", metadata=meta or {"format": "pt"})
        assert done == len(mods), (done, len(mods))
        assert changed > 0, "the adapter changed nothing"
        for name in os.listdir(base):  # config, tokenizer, chat template, processor files: the base's own
            if not name.endswith(".safetensors"):
                shutil.copy(f"{base}/{name}", f"{out}/{name}")
        print(f"merged {done} layers (scale {scale}, largest weight change {changed:.4g})")
        ''')

    def hub_json(path):  # a JSON file from hadron-data, or None if it isn't there
        try:
            return json.load(open(hf_hub_download(DATASET, path, repo_type="dataset", token=tok, local_dir=work,
                                                  force_download=True), encoding="utf-8"))
        except Exception as e:
            if "404" in str(e) or "EntryNotFound" in type(e).__name__:
                return None
            raise

    def upload(path, obj, msg):
        api.upload_file(path_or_fileobj=json.dumps(obj).encode("utf-8"), path_in_repo=path, repo_id=DATASET,
                        repo_type="dataset", commit_message=msg)

    def run_model(name):
        REPO, STEP, NEED_GB, EXTRA = MODELS[name]
        TAG = (f"{REPO.split('/')[-1]}-step{STEP}" if STEP else REPO.split("/")[-1]) + "-32k"
        outs = {f: f"gen/{TAG}/{f}.json" for f in FILES}
        done_before = {f: hub_json(outs[f]) for f in FILES}
        if all(d and d.get("complete") for d in done_before.values()):
            log(f"== {TAG}: answers already uploaded, skipped")
            return TAG
        log(f"== {TAG}")
        stop_beat = threading.Event()

        def beat():  # progress.txt on the Hub every 2 minutes while this model runs, when there's a new log line
            sent = [-1]
            while not stop_beat.wait(120):
                if len(log_lines) == sent[0]:
                    continue
                sent[0] = len(log_lines)
                try:
                    text = "\n".join(log_lines[-60:] + ["", f"stage: {stage[0]}", machine()])
                    api.upload_file(path_or_fileobj=text.encode(), repo_id=DATASET, repo_type="dataset",
                                    path_in_repo=f"gen/{TAG}/progress.txt", commit_message=f"{TAG} progress")
                except Exception as e:
                    print("(progress upload failed:", str(e)[:200] + ")", flush=True)

        threading.Thread(target=beat, daemon=True).start()
        server, merged = None, None
        try:
            # The model: a plain one straight from the Hub (pinned to today's commit), or a Hadron checkpoint merged
            # into its base.
            if STEP:
                stage[0] = "merging the Hadron checkpoint"
                rev = {c.title: c.commit_id for c in api.list_repo_commits(REPO)}.get(f"checkpoint step {STEP}")
                assert rev, f"No 'checkpoint step {STEP}' commit in {REPO}"
                merged = f"{work}/merged-{TAG}"
                if not os.path.exists(f"{merged}/.ok"):
                    shutil.rmtree(merged, ignore_errors=True)
                    adapter = snapshot_download(REPO, revision=rev, token=tok, local_dir=f"{work}/adapter-{TAG}",
                                                allow_patterns=["last-checkpoint/adapter_*"])
                    base_dir = snapshot_download(BASE, revision=BASE_REV, token=tok)
                    open(f"{work}/merge.py", "w").write(MERGE)
                    log(sh(f'{venv}/bin/python {work}/merge.py "{adapter}/last-checkpoint" "{base_dir}" "{merged}"').strip())
                    open(f"{merged}/.ok", "w").close()
                    shutil.rmtree(f"{work}/adapter-{TAG}", ignore_errors=True)
                serve = [merged]
            else:
                rev = BASE_REV if REPO == BASE else api.model_info(REPO).sha
                serve = [REPO, "--revision", rev, "--tokenizer-revision", rev, *EXTRA]
            log(f"{REPO}{f' step {STEP}' if STEP else ''} @ {rev[:7]}")

            free = [float(x) / 1024 for x in sh("nvidia-smi --query-gpu=memory.free,memory.total "
                                                "--format=csv,noheader,nounits").split(",")]
            assert free[0] >= NEED_GB + 4, (f"{TAG} needs ~{NEED_GB} GB of GPU memory but only {free[0]:.0f} GB is free: "
                                            "is another notebook using this GPU? Restart this one and try again.")
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            stage[0] = "starting vLLM (downloads the model the first time)"
            vlog = f"{work}/vllm-{TAG}.log"
            server = subprocess.Popen([f"{venv}/bin/vllm", "serve", *serve, "--served-model-name", TAG,
                                       "--port", str(port), "--max-model-len", str(CTX),
                                       "--gpu-memory-utilization", str(round(min(0.95, max(0.85, (NEED_GB + 4) / gtotal)), 2)),
                                       "--max-num-seqs", str(CONC),
                                       "--reasoning-parser", "qwen3"],
                                      env=dict(os.environ, HF_TOKEN=tok, VLLM_USE_FLASHINFER_SAMPLER="0"),
                                      stdout=open(vlog, "w"), stderr=subprocess.STDOUT, start_new_session=True)
            with open(pidfile, "a") as f:
                f.write(f"{server.pid}\n")
            t0 = said = time.time()
            while True:
                if server.poll() is not None:
                    raise RuntimeError("vLLM exited while loading:\n" + open(vlog, errors="replace").read()[-3000:])
                try:
                    urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5)
                    break
                except Exception:
                    if time.time() - t0 > 30 * 60:
                        raise RuntimeError("vLLM did not start in 30 min:\n" + open(vlog, errors="replace").read()[-3000:])
                    if time.time() - said > 120:
                        said = time.time()
                        tail = [l for l in open(vlog, errors="replace").read().splitlines() if l.strip()][-1:]
                        log(f"vLLM loading ({time.time() - t0:.0f}s): {tail[0][-160:] if tail else ''} | {machine()}")
                    time.sleep(10)
            log(f"vLLM serving {TAG} after {time.time() - t0:.0f}s | {machine()}")

            url = f"http://localhost:{port}/v1/chat/completions"

            def ask(task):  # the request eval.py's query_endpoint sends, keeping how it ended and its length
                body = json.dumps({"model": TAG, "temperature": TEMP, "messages": [
                    {"role": "system", "content": system_prompt_for(task)},
                    {"role": "user", "content": task_prompt(task)}]}).encode("utf-8")
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=1800) as r:
                        out = json.loads(r.read().decode("utf-8"))
                    c = out["choices"][0]
                    return [c["message"].get("content") or "", c.get("finish_reason"),
                            (out.get("usage") or {}).get("completion_tokens", 0)]
                except urllib.error.HTTPError as e:
                    if e.code == 400:  # e.g. the prompt is longer than the context: the same every time, don't retry
                        return ["", f"rejected: {e.read().decode('utf-8', 'replace')[:200]}", 0]
                    return ["", f"error: {str(e)[:200]}", 0]
                except Exception as e:
                    return ["", f"error: {str(e)[:200]}", 0]

            for f in FILES:
                prev = done_before[f] or {}
                if prev.get("complete"):
                    continue
                answers = prev.get("answers", {})  # task id -> [[content, finish_reason, tokens], ...]
                answers = {k: v for k, v in answers.items() if not any(str(a[1]).startswith("error") for a in v)}
                todo = [t for t in tasks[f] if len(answers.get(t["id"], [])) < N]
                stage[0] = f"answering {f}"
                log(f"-- {f}: {len(todo)} of {len(tasks[f])} tasks to answer ({len(tasks[f]) - len(todo)} done before)")
                lock, last_up, t1, uploads = threading.Lock(), [time.time()], time.time(), []
                key = {"model": REPO, "step": STEP, "rev": rev, "data": f, "n": N, "temperature": TEMP, "ctx": CTX}

                def one(task):  # one answer; a task's 4 answers run side by side, so slow tasks don't drag on
                    got = ask(task)
                    with lock:
                        answers[task["id"]] = answers.get(task["id"], []) + [got]
                        k = sum(len(v) == N for v in answers.values())
                        if len(answers[task["id"]]) == N and k % 20 == 0:
                            log(f"{f}: {k}/{len(tasks[f])} tasks answered ({time.time() - t1:.0f}s) | {machine()}")
                        if time.time() - last_up[0] > 300:  # checkpoint: a crash loses at most ~5 minutes
                            last_up[0] = time.time()
                            snap = dict(key, complete=False, answers=dict(answers))
                            th = threading.Thread(target=upload, args=(outs[f], snap, f"{TAG}: answers so far"),
                                                  daemon=True)
                            th.start()
                            uploads.append(th)

                jobs = [t for t in todo for _ in range(N - len(answers.get(t["id"], [])))]
                with ThreadPoolExecutor(CONC) as pool:  # = vLLM's --max-num-seqs
                    list(pool.map(one, jobs))
                for th in uploads:  # a late "so far" upload must not land after the complete one
                    th.join()
                flat = [a for v in answers.values() for a in v]
                empty = sum(not a[0].strip() for a in flat)
                cut = sum(a[1] == "length" for a in flat)
                errs = sum(str(a[1]).startswith("error") for a in flat)
                toks = sorted(a[2] for a in flat)
                log(f"{f}: {len(flat)} answers in {time.time() - t1:.0f}s | empty {empty}, cut off at {CTX // 1024}k "
                    f"{cut}, failed requests {errs} | median {toks[len(toks) // 2]} tokens, max {toks[-1]}")
                assert errs < 0.05 * len(flat), "Too many failed requests: not marking these answers complete"
                upload(outs[f], dict(key, complete=True, answers=answers), f"{TAG}: answers for {f}")
                log(f"uploaded -> {DATASET}/{outs[f]}")
            return TAG
        except Exception as e:
            log(f"FAILED ({stage[0]}): {str(e)[-1500:]}")
            raise
        finally:
            stop_beat.set()
            if server:
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(10)
            try:
                api.upload_file(path_or_fileobj="\n".join(log_lines[-80:] + ["", f"stage: {stage[0]}", machine()]).encode(),
                                repo_id=DATASET, repo_type="dataset", path_in_repo=f"gen/{TAG}/progress.txt",
                                commit_message=f"{TAG} progress")
            except Exception:
                pass
            # the next model needs the disk: each download is 9-72 GB
            gone = [os.path.join(HF_HUB_CACHE, "models--" + REPO.replace("/", "--"))] + ([merged] if merged else [])
            before = shutil.disk_usage(work).used
            for d in gone:
                size = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(d) for x in fs) if os.path.isdir(d) else 0
                shutil.rmtree(d, ignore_errors=True)
                log(f"deleted {d}: {size / 1e9:.1f} GB ({'still there!' if os.path.exists(d) else 'gone'})")
            log(f"disk used {before / 1e9:.0f} -> {shutil.disk_usage(work).used / 1e9:.0f} GB")

    finished, failed = [], []
    for name in choice.value:
        try:
            finished.append(run_model(name))
        except Exception as e:  # one model failing shouldn't stop the others
            print(f"!! {name} failed: {str(e)[-1500:]}", flush=True)
            failed.append(name)
    print(f"\nDone. Answers ready for: {', '.join(finished) or 'none'}"
          + (f". Failed: {', '.join(failed)}" if failed else "")
          + "\nNext: tell Claude to score them (hadron_score.py runs the tests off molab).", flush=True)
    return


if __name__ == "__main__":
    app.run()
