"""The real-repo FIM benchmark's tests on this Linux machine (e.g. molab's free CPUs) instead of Modal containers.

Same scoring as `modal_repo_fim.py --eval`: an empty fill fails, a fill identical to the original passes, every other
fill is spliced into the real file at the repo's pinned commit and that repo's own tests decide. Needs no root: Node 24
(the official build), GCC 14, CMake, Ninja, git and uv come from nodejs.org / conda-forge into --tools.

  python local_repo_eval.py --data repo_holes_ctx2.jsonl --fills repo_eval_report_q12v07.fills.json \
      --report repo_eval_report_ctx2.jsonl
  (--fills is modal_repo_fim's fills cache: {"key": {...}, "fills": {model: [fill per hole]}}; --limit 3 for a smoke
  test; --repos owner/name,... for some repos only)

The report keeps every earlier model's results, so the table shows the new model next to them.
"""
import argparse
import collections
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import repo_fim  # noqa: E402


def _download(url: str, path: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "neutron-bench"})
    with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)
    return path


def setup_toolchain(tools: str, log=print) -> None:
    """Install the test toolchain under `tools` (once) and point this process's environment at it, so every
    repo_fim.run() subprocess sees it. Matches modal_repo_fim's image: Node 24 + corepack, GCC 14, CMake, Ninja, uv."""
    os.makedirs(tools, exist_ok=True)
    node, env, shims = f"{tools}/node", f"{tools}/env", f"{tools}/shims"
    if not os.path.exists(f"{node}/bin/node"):
        base = "https://nodejs.org/dist/latest-v24.x/"
        sums = urllib.request.urlopen(urllib.request.Request(base + "SHASUMS256.txt",
                                                             headers={"User-Agent": "neutron-bench"})).read().decode()
        name = re.search(r"node-v24[\d.]+-linux-x64\.tar\.xz", sums).group(0)
        log(f"[tools] {name}")
        tarfile.open(_download(base + name, f"{tools}/node.tar.xz")).extractall(tools, filter="tar")
        os.rename(f"{tools}/{name[:-len('.tar.xz')]}", node)
        os.remove(f"{tools}/node.tar.xz")
    if not os.path.exists(f"{env}/bin/cmake"):
        log("[tools] micromamba: GCC 14, CMake, Ninja, git, uv from conda-forge (a few minutes)")
        mm = f"{tools}/micromamba"
        tarfile.open(_download("https://micro.mamba.pm/api/micromamba/linux-64/latest", f"{mm}.tar.bz2")).extractall(mm, filter="tar")
        r = subprocess.run([f"{mm}/bin/micromamba", "create", "-y", "-q", "-p", env, "-c", "conda-forge",
                            "--override-channels", "gcc_linux-64=14", "gxx_linux-64=14", "cmake", "ninja", "git", "uv",
                            "make", "pkg-config", "python=3.12"],  # 3.12: the Python uv found on Modal
                           env=dict(os.environ, MAMBA_ROOT_PREFIX=f"{tools}/mamba"), capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("micromamba failed:\n" + (r.stdout + r.stderr)[-3000:])
        shutil.rmtree(f"{tools}/mamba/pkgs", ignore_errors=True)  # the downloaded packages (~1 GB): installed now
        os.remove(f"{mm}.tar.bz2")
    os.makedirs(shims, exist_ok=True)  # plain names for build scripts that don't read CC / CXX (conda's are prefixed)
    for short, full in (("gcc", "gcc"), ("cc", "gcc"), ("gcc-14", "gcc"), ("g++", "g++"), ("c++", "g++"),
                        ("g++-14", "g++"), ("ar", "ar"), ("ranlib", "ranlib"), ("ld", "ld"), ("nm", "nm"),
                        ("strip", "strip")):
        if not os.path.lexists(f"{shims}/{short}"):
            os.symlink(f"{env}/bin/x86_64-conda-linux-gnu-{full}", f"{shims}/{short}")
    home = f"{tools}/home"  # caches go here: the notebook user's own home may not be writable
    os.makedirs(home, exist_ok=True)
    os.environ.update({
        "PATH": f"{shims}:{node}/bin:{env}/bin:" + os.environ.get("PATH", ""),
        "CC": f"{env}/bin/x86_64-conda-linux-gnu-gcc", "CXX": f"{env}/bin/x86_64-conda-linux-gnu-g++",
        # conda's GCC 14 brings a newer libstdc++ than the system's: test binaries must load that one
        "LDFLAGS": f"-Wl,-rpath,{env}/lib", "HOME": home, "XDG_CACHE_HOME": f"{home}/.cache",
        "XDG_DATA_HOME": f"{home}/.local/share", "npm_config_cache": f"{home}/.npm", "COREPACK_HOME": f"{home}/corepack",
        "UV_CACHE_DIR": f"{home}/.cache/uv", "UV_PYTHON_INSTALL_DIR": f"{home}/uv-python",
        # the rest as in modal_repo_fim's image
        "COREPACK_ENABLE_DOWNLOAD_PROMPT": "0", "CI": "1", "UV_LINK_MODE": "copy", "TZ": "UTC", "UV_TORCH_BACKEND": "cpu",
        # C++ builds: 2 compile jobs each, since several repos build at once (Ninja would otherwise start one per
        # core it can see, which in a container can be the whole host's)
        "CMAKE_BUILD_PARALLEL_LEVEL": "2",
    })
    for k in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
        os.environ.pop(k, None)
    if not os.path.exists(f"{node}/bin/pnpm"):
        ok, _, tail = repo_fim.run("corepack enable", tools, 120)
        if not ok:
            raise RuntimeError(f"corepack enable failed:\n{tail}")
    ok, _, tail = repo_fim.run("node --version && g++ --version | head -1 && cmake --version | head -1 && "
                               "uv --version && git --version", tools, 120)
    if not ok:
        raise RuntimeError(f"toolchain check failed:\n{tail}")
    log("[tools] " + " | ".join(tail.splitlines()))


def _checkout(info: dict, root: str) -> None:
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    ok, _, tail = repo_fim.run(f"git init -q && git remote add origin {info['url']} && "
                               f"git fetch -q --depth 1 origin {info['commit']} && git checkout -q FETCH_HEAD",
                               root, 600)
    if not ok:
        raise RuntimeError(f"clone failed: {tail}")


def _check_batch(repo: str, info: dict, fills: list[tuple[str, int, int, str]], root: str, log) -> list[bool | None]:
    """modal_repo_fim.check on this machine: install the repo, confirm its tests pass, then test each fill.
    A repo that won't install or pass its own tests gives None ("couldn't test") for its fills."""
    # This batch's own package caches (npm, pnpm, uv), next to the checkout and deleted with it: shared caches kept
    # growing, and on molab the disk lives in RAM, so they ate into the notebook's memory until it was killed.
    c = f"{root}.cache"
    env = (f"export npm_config_cache={c}/npm npm_config_store_dir={c}/pnpm-store XDG_DATA_HOME={c}/data "
           f"XDG_CACHE_HOME={c}/cache UV_CACHE_DIR={c}/uv; ")
    cfg = {k: env + v if k in ("setup", "test") else v for k, v in repo_fim.REPOS[repo].items()}
    try:
        _checkout(info, root)
        ok, _, tail = repo_fim.run(cfg["setup"], root, 30 * 60)
        if not ok:
            raise RuntimeError(f"setup failed:\n{tail}")
        for _ in range(3):  # a flaky test shouldn't sink a whole batch
            ok, secs, tail = repo_fim.run(cfg["test"], root, 15 * 60)
            if ok:
                break
        if not ok:
            raise RuntimeError(f"the original code's tests fail:\n{tail}")
        timeout = max(60.0, 3 * secs + 30)
        out = [repo_fim.check_fill(root, path, lo, hi, middle, cfg["test"], timeout)[0]
               for path, lo, hi, middle in fills]
        log(f"[{repo}] {sum(out)}/{len(out)} fills pass")
        return out
    except Exception as e:
        log(f"!! [{repo}] {len(fills)} fills could NOT be tested: {e}")
        return [None] * len(fills)
    finally:
        n = _kill_inside(root)
        if n:
            log(f"[{repo}] stopped {n} leftover process(es)")
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(c, ignore_errors=True)


def _kill_inside(root: str) -> int:
    """Kill processes still running inside `root` (their working directory is there): tests that started a server
    or a worker in a new session escape repo_fim.run's process-group kill. Returns how many."""
    import signal

    n = 0
    for pid in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
            if cwd == root or cwd.startswith(root + "/"):
                os.kill(int(pid), signal.SIGKILL)
                n += 1
        except OSError:
            pass
    return n


MEM_GB = 0.0  # --mem-gb: the real memory allowance, when the machine can't see it (molab shows the host's RAM)


def _machine() -> tuple[float, float, str]:
    """(memory used GB, memory limit GB, short status) of this container, for the progress lines: if the notebook
    dies, the last line shows what was running out. With MEM_GB set, files in /tmp count as memory too: molab's
    disk lives in RAM (its size reads as ~9 billion GB)."""
    def read(path):
        try:
            return open(path).read().strip()
        except OSError:
            return ""

    v2 = bool(read("/sys/fs/cgroup/memory.current"))
    cg = "/sys/fs/cgroup/" if v2 else "/sys/fs/cgroup/memory/memory."
    used = read(cg + ("memory.current" if v2 else "usage_in_bytes"))
    limit = read(cg + ("memory.max" if v2 else "limit_in_bytes"))
    stat = dict(line.split() for line in read(cg + ("memory.stat" if v2 else "stat")).splitlines()
                if len(line.split()) == 2)
    cache = int(stat.get("inactive_file" if v2 else "total_inactive_file", 0))  # file cache the kernel can drop
    info = dict(line.split(":", 1) for line in read("/proc/meminfo").splitlines() if ":" in line)
    total = int(info.get("MemTotal", "0 kB").split()[0]) * 1024
    avail = int(info.get("MemAvailable", "0 kB").split()[0]) * 1024
    used_b = int(used) - cache if used.isdigit() else total - avail
    limit_b = int(limit) if limit.isdigit() and int(limit) < total else total
    procs = sum(p.isdigit() for p in os.listdir("/proc")) if os.path.isdir("/proc") else 0
    tmp = shutil.disk_usage("/tmp").used if os.path.isdir("/tmp") else 0
    if MEM_GB:
        used_b, limit_b = used_b + tmp, MEM_GB * 1e9
    status = (f"memory {used_b / 1e9:.1f}/{limit_b / 1e9:.0f} GB" + (" (RAM + /tmp files)" if MEM_GB else "")
              + f" | /tmp {tmp / 1e9:.1f} GB | {procs} processes")
    return used_b / 1e9, limit_b / 1e9, status


def _hub(spec: str, local: str, upload: bool) -> None:
    """Copy the progress file to / from "<dataset>:<path>" on Hugging Face (token: HF_TOKEN), so a rerun can
    continue even after the machine's disk was wiped. Failures are reported, never fatal."""
    from huggingface_hub import HfApi, hf_hub_download

    repo_id, path = spec.split(":", 1)
    try:
        if upload:
            HfApi().upload_file(path_or_fileobj=local, path_in_repo=path, repo_id=repo_id, repo_type="dataset",
                                commit_message="benchmark progress")
        else:
            got = hf_hub_download(repo_id, path, repo_type="dataset", local_dir=os.path.dirname(local) or ".",
                                  force_download=True)
            if os.path.abspath(got) != os.path.abspath(local):
                shutil.copyfile(got, local)
            print(f"progress from an earlier run: {spec}", flush=True)
    except Exception as e:
        if upload or "404" not in str(e) and "EntryNotFound" not in type(e).__name__:
            print(f"(progress {'upload' if upload else 'download'} skipped: {str(e)[:200]})", flush=True)


def evaluate(rows: list[dict], fills: dict[str, list[str]], info: dict, work: str, workers: int = 4, chunk: int = 12,
             progress: str = "", hub_progress: str = "", log=print) -> dict[tuple[str, int], bool | None]:
    """(model, hole index) -> passed (None: couldn't be tested). Batches of `chunk` fills from one repo run in
    parallel, each in its own fresh checkout, slowest repos first. Each finished batch is appended to `progress`
    (if given), and a rerun skips the fills already tested there, so a restarted notebook continues."""
    done_before = {}
    if progress and hub_progress and not os.path.exists(progress):
        _hub(hub_progress, progress, upload=False)
    if progress and os.path.exists(progress):
        for line in open(progress, encoding="utf-8"):
            x = json.loads(line)
            if x["pass"] is not None:
                done_before[x["model"], x["id"], x["fill"]] = x["pass"]
    passed, jobs = {}, collections.defaultdict(list)
    for model, mids in fills.items():
        for i, (r, mid) in enumerate(zip(rows, mids)):
            if not mid.strip():
                passed[model, i] = False
            elif repo_fim.same_code(mid, r["oracle_middle"]):
                passed[model, i] = True
            elif (model, r["id"], mid) in done_before:
                passed[model, i] = done_before[model, r["id"], mid]
            else:
                jobs[r["repo"]].append((model, i, (r["path"], r["lo"], r["hi"], mid)))
    batches = [(repo, js[k:k + chunk]) for repo, js in jobs.items() for k in range(0, len(js), chunk)]
    slow = {"EbodShojaei/bake": 54, "MichielMe/fastscheduler": 27, "meodai/heerich": 12}  # test seconds, from mining
    batches.sort(key=lambda b: -len(b[1]) * slow.get(b[0], 8))
    if done_before:
        log(f"{len(done_before)} fills were already tested in an earlier run (from {progress})")
    log(f"{sum(passed.values())} fills pass without a new test run (exact or tested before); testing "
        f"{sum(len(b) for _, b in batches)} others in {len(batches)} batches, {workers} at a time")
    t0, done, lock = time.time(), [0], threading.Lock()
    # One CPU per batch: test runners such as vitest start a worker per CPU they can see, and in a container that can
    # be every core of the host machine, so 4 batches at once could start dozens of workers and freeze the notebook.
    # Linux's affinity is per thread, and the tests started from this thread inherit it.
    cpus = queue.Queue()
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [None]
    for k in range(workers):
        cpus.put(allowed[k % len(allowed)])

    running = [0]

    def run(k):
        repo, b = batches[k]
        cpu = cpus.get()
        waited = 0
        while True:  # don't start another batch while memory is nearly full: wait for running ones to finish
            used, limit, status = _machine()
            with lock:
                if not limit or used < 0.75 * limit or running[0] == 0:
                    running[0] += 1
                    break
            if waited % 120 == 0:
                log(f"   (memory {used:.1f}/{limit:.0f} GB: {repo} waits for a running batch to finish)")
            time.sleep(10)
            waited += 10
        try:
            if cpu is not None:
                os.sched_setaffinity(0, {cpu})
            res = _check_batch(repo, info[repo], [j[2] for j in b], f"{work}/b{k}_{repo.replace('/', '__')}", log)
        finally:
            cpus.put(cpu)
            with lock:
                running[0] -= 1
        with lock:
            done[0] += 1
            if progress:
                with open(progress, "a", encoding="utf-8") as f:
                    f.writelines(json.dumps({"model": m, "id": rows[i]["id"], "fill": j[3], "pass": ok}) + "\n"
                                 for (m, i, j), ok in zip(b, res))
                if hub_progress:
                    _hub(hub_progress, progress, upload=True)
        log(f"   {done[0]}/{len(batches)} batches done ({(time.time() - t0) / 60:.0f} min) | {_machine()[2]}")
        return res

    with ThreadPoolExecutor(workers) as pool:
        for (repo, b), res in zip(batches, pool.map(run, range(len(batches)))):
            for (model, i, _), ok in zip(b, res):
                passed[model, i] = ok
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the holes, e.g. repo_holes_ctx2.jsonl")
    ap.add_argument("--fills", required=True, help="fills cache JSON: {'key': ..., 'fills': {model: [...]}}")
    ap.add_argument("--report", required=True, help="per-hole report to merge the results into")
    ap.add_argument("--repo-info", default=os.path.join(HERE, "quark_repos.jsonl"))
    ap.add_argument("--tools", default="/tmp/quark_tools")
    ap.add_argument("--work", default="/tmp/quark_bench")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--repos", default="")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N holes of each repo")
    ap.add_argument("--show", type=int, default=0, help="table: only the best N earlier models (0: all)")
    ap.add_argument("--progress", default="", help="file of finished batches: a rerun continues from it")
    ap.add_argument("--hub-progress", default="",
                    help="<dataset>:<path> on Hugging Face to keep --progress in too (survives a wiped disk; HF_TOKEN)")
    ap.add_argument("--mem-gb", type=float, default=0,
                    help="memory allowance in GB, counting /tmp files too (molab: its disk lives in RAM)")
    args = ap.parse_args()
    global MEM_GB
    MEM_GB = args.mem_gb
    log = lambda m: print(m, flush=True)
    if hasattr(os, "nice"):
        os.nice(10)  # the tests yield to the notebook, so it stays responsive while they run

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    saved = json.load(open(args.fills, encoding="utf-8"))
    if saved["key"]["ids"] != [r["id"] for r in rows]:
        raise SystemExit(f"{args.fills} was made for other holes than {args.data}")
    keep = list(range(len(rows)))
    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    if repos:
        keep = [i for i in keep if rows[i]["repo"] in repos]
    if args.limit:
        per = collections.Counter()
        keep = [i for i in keep if (per.update([rows[i]["repo"]]) or per[rows[i]["repo"]] <= args.limit)]
    rows = [rows[i] for i in keep]
    fills = {m: [f[i] for i in keep] for m, f in saved["fills"].items()}
    log(f"Real-repo FIM benchmark on this machine: {len(fills)} model(s) x {len(rows)} holes, "
        f"{len({r['repo'] for r in rows})} repos")

    setup_toolchain(args.tools, log)
    log("machine: " + _machine()[2])
    t0 = time.time()
    passed = evaluate(rows, fills, repo_fim.load_repo_info(args.repo_info), args.work, args.workers,
                      progress=args.progress, hub_progress=args.hub_progress, log=log)
    untested = sum(v is None for v in passed.values())
    log(f"tests took {(time.time() - t0) / 60:.0f} min")

    records = [{"model": m, "id": r["id"], "repo": r["repo"], "lang": r["lang"], "size": r["size"],
                "pass": bool(passed[m, i]), "completion": fills[m][i]} for m in fills for i, r in enumerate(rows)]
    repo_fim.print_tables(repo_fim.merge_report(args.report, records), list(fills), args.show)
    if untested:
        log(f"\n!! {untested} fills could not be tested (see the '!!' lines above) and count as failed: the score is "
            f"too low until those repos work here")
    log(f"\nPer-hole results -> {args.report}")


if __name__ == "__main__":
    main()
