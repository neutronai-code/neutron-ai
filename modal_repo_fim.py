"""Quark on real repos, on Modal (see repo_fim.py): mine holes, then benchmark models on them.

Mining: one cheap CPU container per repo, all in parallel.
  modal run modal_repo_fim.py                                   # the 10 repos in repo_fim.REPOS
  modal run modal_repo_fim.py --repos raulcd/datanomy --max-holes 5 --minutes 10   # quick smoke test
  -> repo_holes.jsonl (the holes) and repo_holes_stats.json (per-repo setup/test times and counts)

Benchmark: a GPU writes every model's fill for every hole (greedy, like the other FIM evals), then CPU containers
splice each fill into the real file and run that repo's own tests. A fill identical to the original passes without
running the tests (the original is known to pass).
  modal run modal_repo_fim.py --eval --models Qwen/Qwen2.5-Coder-1.5B,neutronai-code/Quark-1.5B-v0.2
  -> repo_eval_report.jsonl (one line per model x hole) and a table by language / repo / hole size
  Results merge into the existing report, so scoring one new model also shows the earlier ones in the table.
  Add --repos owner/name,... to re-run only those repos. --gen-gpu L40S for models over ~8B (A10G: 24 GB).
  With repo context (make_ctx_bench.py): --data repo_holes_ctx.jsonl --report repo_eval_report_ctx.jsonl

GRPO reward server (RepoChecker): `modal deploy modal_repo_fim.py` once (again after changing repo_fim.py or this
file); `modal_train.py --mode repo-grpo` then sends each fill to the deployed app and gets back pass/fail.
"""
import collections
import json
import os
import pathlib

import modal

HERE = pathlib.Path(__file__).parent
HF_SECRET = os.environ.get("NEUTRON_HF_SECRET", "huggingface")  # same as modal_train.py: the secret holding HF_TOKEN

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("git", "curl", "ca-certificates", "build-essential", "gcc-14", "g++-14", "make")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_24.x | bash -",
        "apt-get install -y nodejs",
        "corepack enable",
    )
    .pip_install("uv", "cmake", "ninja")  # pip's cmake is newer than Ubuntu's 3.28 (Hexi needs 3.29)
    .apt_install("ffmpeg", "libssl-dev", "libcurl4-openssl-dev", "pkg-config")  # second batch: trackio, C++ repos
    .env({"CC": "gcc-14", "CXX": "g++-14", "COREPACK_ENABLE_DOWNLOAD_PROMPT": "0", "CI": "1",
          "UV_LINK_MODE": "copy", "TZ": "UTC", "UV_TORCH_BACKEND": "cpu"})  # CPU torch: no 2 GB of CUDA wheels
)
gpu_image = (  # same layers as modal_fim_eval.py, so Modal reuses that cached build
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("curl", "ca-certificates", "g++", "build-essential", "sqlite3", "git")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_24.x | bash -",
        "apt-get install -y nodejs",
    )
    .uv_pip_install("vllm==0.33.0", "huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
for f in ("fim.py", "repomap.py", "sandbox.py", "repo_fim.py", "modal_fim_eval.py"):
    image = image.add_local_file(HERE / f, f"/app/{f}")
    gpu_image = gpu_image.add_local_file(HERE / f, f"/app/{f}")
image = image.add_local_file(HERE / "quark_repos.jsonl", "/app/quark_repos.jsonl")  # pinned commits, for RepoChecker

app = modal.App("neutron-repo-fim", image=image)
hf_cache = modal.Volume.from_name("neutron-hf-cache", create_if_missing=True)


def _checkout(repo: str, info: dict) -> str:
    """Clone `repo` at its pinned commit into /work; returns the path (raises on failure)."""
    import subprocess

    root = f"/work/{repo.replace('/', '__')}"
    clone = (f"mkdir -p {root} && cd {root} && git init -q && git remote add origin {info['url']} "  # origin: for
             f"&& git fetch -q --depth 1 origin {info['commit']} && git checkout -q FETCH_HEAD")  # relative submodules
    r = subprocess.run(["bash", "-c", clone], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"clone of {repo} failed: {r.stderr[-1000:]}")
    return root


@app.function(cpu=2, memory=4096, timeout=3 * 3600)
def mine(repo: str, info: dict, max_holes: int, minutes: float) -> dict:
    import sys

    sys.path.insert(0, "/app")
    import repo_fim

    try:
        root = _checkout(repo, info)
    except RuntimeError as e:
        return {"rows": [], "stats": {"repo": repo, "status": "clone failed"}, "tail": str(e)}
    cfg = repo_fim.REPOS[repo]
    return repo_fim.mine_repo(repo, root, cfg["setup"], cfg["test"], cfg["src"], info, max_holes, minutes,
                              log=lambda m: print(m, flush=True))


@app.function(gpu="A10G", timeout=2 * 3600, image=gpu_image, secrets=[modal.Secret.from_name(HF_SECRET)],
              volumes={"/root/.cache/huggingface": hf_cache})
def generate(models: list[str], prompts: list[str], max_tokens: int = 512) -> dict[str, list[str]]:
    """Greedy FIM fills for every prompt from every model (one vLLM server at a time)."""
    import os
    import sys
    from concurrent.futures import ThreadPoolExecutor

    os.chdir("/app")
    sys.path.insert(0, "/app")
    import fim
    from modal_fim_eval import _serve, _stop

    out = {}
    for model in models:
        try:
            server = _serve(model)
        except Exception as e:  # one model that won't load (e.g. an architecture this vLLM lacks) mustn't sink the rest
            print(f"!! {model} could not be served ({e}); its fills are left empty", flush=True)
            out[model] = [""] * len(prompts)
            continue
        try:
            hf_cache.commit()
            url = "http://localhost:8000/v1/completions"

            def one(p):
                try:
                    return fim.extract_fim_middle(fim.query_fim_samples(url, model, p, max_tokens=max_tokens)[0])
                except Exception as e:  # count a server error as an empty (failed) fill, but say so
                    print(f"[{model}] request failed: {e}", flush=True)
                    return ""

            with ThreadPoolExecutor(32) as pool:
                out[model] = list(pool.map(one, prompts))
            print(f"{model}: {len(prompts)} fills written", flush=True)
        finally:
            _stop(server)
    return out


@app.function(cpu=2, memory=4096, timeout=3 * 3600)
def check(repo: str, info: dict, fills: list[tuple[str, int, int, str]]) -> list[bool]:
    """Install the repo once, then splice each (path, lo, hi, middle) fill in and run its tests: pass/fail each."""
    import sys

    sys.path.insert(0, "/app")
    import repo_fim

    cfg = repo_fim.REPOS[repo]
    root = _checkout(repo, info)
    ok, _, tail = repo_fim.run(cfg["setup"], root, 30 * 60)
    if not ok:
        raise RuntimeError(f"{repo} setup failed:\n{tail}")
    for _ in range(3):  # a flaky test shouldn't sink a whole batch
        ok, secs, tail = repo_fim.run(cfg["test"], root, 15 * 60)
        if ok:
            break
    if not ok:
        raise RuntimeError(f"{repo} baseline tests fail:\n{tail}")
    timeout = max(60.0, 3 * secs + 30)
    results = []
    for path, lo, hi, middle in fills:
        results.append(repo_fim.check_fill(root, path, lo, hi, middle, cfg["test"], timeout)[0])
    print(f"[{repo}] {sum(results)}/{len(results)} fills pass", flush=True)
    return results


@app.cls(cpu=2, memory=4096, timeout=30 * 60, scaledown_window=90, max_containers=16)
class RepoChecker:
    """GRPO reward server: each container installs one repo once (the next block's repo is warmed up ahead of time),
    then tests one fill at a time. A container whose repo won't install or pass its own tests answers None ("can't
    tell") rather than failing every fill."""
    repo: str = modal.parameter()

    @modal.enter()
    def setup(self):
        import sys

        sys.path.insert(0, "/app")
        import repo_fim

        self.cfg = repo_fim.REPOS[self.repo]
        self.ready = False
        try:
            self.root = _checkout(self.repo, repo_fim.load_repo_info("/app/quark_repos.jsonl")[self.repo])
        except Exception as e:
            print(f"[{self.repo}] checkout failed: {e}", flush=True)
            return
        installed, _, tail = repo_fim.run(self.cfg["setup"], self.root, 30 * 60)
        for _ in range(3 if installed else 0):  # a flaky test shouldn't take the container out
            ok, secs, tail = repo_fim.run(self.cfg["test"], self.root, 15 * 60)
            if ok:
                self.timeout, self.ready = max(60.0, 3 * secs + 30), True
                break
        print(f"[{self.repo}] " + ("ready" if self.ready else "NOT ready:\n" + tail[-2000:]), flush=True)

    @modal.method()
    def warm(self) -> bool:
        return self.ready

    @modal.method()
    def score(self, path: str, lo: int, hi: int, middle: str) -> bool | None:
        import repo_fim

        if not self.ready:
            return None
        return repo_fim.check_fill(self.root, path, lo, hi, middle, self.cfg["test"], self.timeout)[0]


def run_eval(models: list[str], data: str, chunk: int, limit: int, repos: list[str], report: str, gen_gpu: str):
    import repo_fim

    info = repo_fim.load_repo_info(str(HERE / "quark_repos.jsonl"))
    rows = [json.loads(line) for line in open(HERE / data, encoding="utf-8") if line.strip()]
    if repos:
        rows = [r for r in rows if r["repo"] in repos]
    if limit:  # smoke test: the first `limit` holes of each repo
        per = collections.Counter()
        rows = [r for r in rows if (per.update([r["repo"]]) or per[r["repo"]] <= limit)]
    print(f"Real-repo FIM benchmark: {len(models)} models x {len(rows)} holes from "
          f"{len({r['repo'] for r in rows})} repos")
    # The fills are saved as soon as the GPU is done, so a run cut off during testing can be resumed without it.
    cache = HERE / (pathlib.Path(report).stem + ".fills.json")
    key = {"data": data, "ids": [r["id"] for r in rows]}
    saved = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    if saved.get("key") == key and all(m in saved["fills"] for m in models):
        fills = saved["fills"]
        print(f"Reusing the saved fills in {cache.name}")
    else:
        fills = generate.with_options(gpu=gen_gpu).remote(models, [r["fim_prompt"] for r in rows])
        cache.write_text(json.dumps({"key": key, "fills": fills}), encoding="utf-8")

    passed = {}  # (model, row index) -> bool
    jobs = collections.defaultdict(list)  # repo -> [(model, row index, (path, lo, hi, middle))]
    for model in models:
        for i, (r, mid) in enumerate(zip(rows, fills[model])):
            if not mid.strip():
                passed[model, i] = False
            elif repo_fim.same_code(mid, r["oracle_middle"]):
                passed[model, i] = True
            else:
                jobs[r["repo"]].append((model, i, (r["path"], r["lo"], r["hi"], mid)))
    n_exact = sum(passed.values())
    batches = [(repo, js[k:k + chunk]) for repo, js in jobs.items() for k in range(0, len(js), chunk)]
    print(f"{n_exact} fills match the original exactly; testing {sum(len(b) for _, b in batches)} others in "
          f"{len(batches)} containers")
    results = list(check.starmap([(repo, info[repo], [j[2] for j in b]) for repo, b in batches],
                                 return_exceptions=True))
    for (repo, b), res in zip(batches, results):
        if isinstance(res, Exception):
            print(f"!! {repo}: a batch of {len(b)} fills could not be tested ({res}); counted as failed")
            res = [False] * len(b)
        for (model, i, _), ok in zip(b, res):
            passed[model, i] = ok

    report = HERE / report
    records = [{"model": model, "id": r["id"], "repo": r["repo"], "lang": r["lang"], "size": r["size"],
                "pass": passed[model, i], "completion": fills[model][i]}
               for model in models for i, r in enumerate(rows)]
    if report.exists():  # keep earlier models' results; replace only the (model, hole) pairs just re-run
        redone = {(x["model"], x["id"]) for x in records}
        old = [json.loads(line) for line in open(report, encoding="utf-8") if line.strip()]
        records = [x for x in old if (x["model"], x["id"]) not in redone] + records
    with open(report, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in records)

    res = collections.defaultdict(dict)  # model -> hole id -> record, over every repo in the report
    for x in records:
        res[x["model"]][x["id"]] = x
    ids = sorted(set.intersection(*(set(res[m]) for m in models)))
    # Show earlier models that were scored on all the same holes too, so one new model is compared at a glance.
    models = [m for m in res if m not in models and set(ids) <= set(res[m])] + models

    def table(title, field):
        groups = sorted({res[models[0]][i][field] for i in ids})
        tags = [m.split("/")[-1] for m in models]
        print(f"\n{title:28}" + "".join(f"{t[:22]:>24}" for t in tags))
        for g in groups + ["ALL"]:
            sel = [i for i in ids if g == "ALL" or res[models[0]][i][field] == g]
            cells = "".join(f"{sum(res[m][i]['pass'] for i in sel) / len(sel):>17.1%} ({len(sel):>3})" for m in models)
            print(f"{str(g)[:28]:28}{cells}")

    table("language", "lang")
    table("hole size", "size")
    table("repo", "repo")
    print(f"\nPer-hole results -> {report.name}")


@app.local_entrypoint()
def main(repos: str = "", max_holes: int = 40, minutes: float = 45, out: str = "repo_holes.jsonl",
         eval: bool = False, models: str = "", data: str = "repo_holes.jsonl", chunk: int = 40, limit: int = 0,
         report: str = "repo_eval_report.jsonl", gen_gpu: str = "A10G"):
    import repo_fim

    if eval:
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        if not model_list:
            raise SystemExit("--eval needs --models, e.g. --models Qwen/Qwen2.5-Coder-1.5B,neutronai-code/Quark-1.5B-v0.2")
        return run_eval(model_list, data, chunk, limit, [r.strip() for r in repos.split(",") if r.strip()], report,
                        gen_gpu)

    info = repo_fim.load_repo_info(str(HERE / "quark_repos.jsonl"))
    names = [r.strip() for r in repos.split(",") if r.strip()] or list(repo_fim.REPOS)
    unknown = [n for n in names if n not in repo_fim.REPOS or n not in info]
    if unknown:
        raise SystemExit(f"No config / info for {unknown}; add them to REPOS in repo_fim.py.")
    print(f"Mining up to {max_holes} holes per repo ({minutes:.0f} min budget each) from {len(names)} repos")
    results = list(mine.starmap([(n, info[n], max_holes, minutes) for n in names], return_exceptions=True))

    rows, stats = [], []
    for name, res in zip(names, results):
        if isinstance(res, Exception):
            stats.append({"repo": name, "status": f"crashed: {type(res).__name__}: {res}"})
            continue
        rows += res["rows"]
        stats.append(res["stats"])
        if res.get("tail"):
            print(f"\n--- {name}: {res['stats'].get('status')} ---\n{res['tail']}")
    with open(HERE / out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (HERE / out).with_name(pathlib.Path(out).stem + "_stats.json").write_text(json.dumps(stats, indent=1))

    print(f"\n{'repo':28} {'status':24} {'test s':>6} {'checked':>7} {'kept':>5} {'short':>5} {'long':>5}")
    for s in stats:
        mine_rows = [r for r in rows if r["repo"] == s["repo"]]
        print(f"{s['repo']:28} {s.get('status', '')[:24]:24} {s.get('test_s', ''):>6} {s.get('checked', ''):>7} "
              f"{len(mine_rows):>5} {sum(r['size'] == 'short' for r in mine_rows):>5} "
              f"{sum(r['size'] == 'long' for r in mine_rows):>5}")
    print(f"\nWrote {len(rows)} holes -> {out}")
