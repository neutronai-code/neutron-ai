"""NeutronAI v0.9 — 5-Language Execution Sandbox (Python + JS + TypeScript + C++23 + SQLite SQL)
with SHA-256 Content-Addressed Verification Cache & Multi-File Repo Verifier."""
import ast, collections, concurrent.futures, hashlib, json, os, re, shutil, sqlite3, subprocess, sys, tempfile, threading, time

CODE_BLOCK = re.compile(
    r"```(?:python|py|typescript|ts|javascript|js|node|cpp|c\+\+|cxx|cc|c|sql|sqlite)?[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
# REPLACE section may be empty (pure deletion); the lazy `??` keeps it from swallowing the next block.
PATCH_BLOCK = re.compile(
    r"<<<<<<+[ \t]*SEARCH[ \t]*\r?\n(.*?)\r?\n======+[ \t]*\r?\n(?:(.*?)\r?\n)??>>>>>>+[ \t]*REPLACE",
    re.DOTALL,
)
FILE_PATCH_BLOCK = re.compile(
    r"(?:^|\n)###\s*FILE:\s*([^\n]+)\n(.*?)(?=\n###\s*FILE:|\Z)",
    re.DOTALL,
)

_NODE_BIN = shutil.which("node")
_GPP_BIN = shutil.which("g++")

_EXTRA_DIRS = [os.path.dirname(b) for b in (_NODE_BIN, _GPP_BIN) if b]
_TS_FLAGS: list[str] | None = None


def _ts_node_flags() -> list[str]:
    """Node 22.7-25 can run full TS (enums, parameter properties) with `--experimental-transform-types`;
    Node 26 dropped the flag and only strips erasable types. Probe once and use it when available."""
    global _TS_FLAGS
    if _TS_FLAGS is None:
        flag = "--experimental-transform-types"
        try:
            ok = bool(_NODE_BIN) and subprocess.run([_NODE_BIN, flag, "-e", "0"], capture_output=True, timeout=15).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        _TS_FLAGS = [flag] if ok else []
    return _TS_FLAGS
_BASE_PATH = os.environ.get("PATH", "")
_SAFE_ENV = {
    k: os.environ[k]
    for k in ("SYSTEMROOT", "WINDIR", "PATHEXT", "TEMP", "TMP")
    if k in os.environ
} | {
    "PATH": os.pathsep.join([*_EXTRA_DIRS, _BASE_PATH]),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
    "NODE_NO_WARNINGS": "1",
}

# Content-addressed LRU cache (speeds up identical GRPO rollouts & C++ compiles by 100x+)
_CACHE_MAX = 4096
_VERIFY_CACHE: collections.OrderedDict[str, tuple[float, dict]] = collections.OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_STATS = {"hits": 0, "misses": 0}

_BANNED_MODULES = frozenset({
    "subprocess", "socket", "urllib", "http", "ftplib", "ctypes", "multiprocessing",
    "threading", "signal", "shutil", "webbrowser", "pty", "resource", "builtins",
    "importlib", "inspect", "gc", "marshal", "pickle", "atexit",
})
_BANNED_ATTRS = frozenset({
    "system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve",
    "spawnvp", "spawnvpe", "execl", "execle", "execlp", "execlpe", "execv", "execve",
    "execvp", "execvpe", "_exit", "kill", "remove", "unlink", "rmdir", "makedirs",
    "__subclasses__", "__globals__", "__builtins__", "__code__", "__closure__", "f_globals", "f_locals",
})
_BANNED_CALLS = frozenset({
    "exit", "quit", "eval", "exec", "compile", "__import__", "open", "input",
    "breakpoint", "getattr", "setattr", "delattr", "globals", "locals", "vars",
})

_JS_TS_BANNED = re.compile(
    r"\b(?:child_process|process\.exit|process\.kill|process\.binding|process\.dlopen|"
    r"require\s*\(\s*['\"](?:node:)?(?:fs|net|http|https|dgram|cluster|worker_threads|vm|inspector|os|dns|tls)['\"]|"
    r"from\s+['\"](?:node:)?(?:fs|net|http|https|child_process|dgram|cluster|worker_threads|vm|inspector|os|dns|tls)['\"]|"
    r"import\s*\(\s*['\"](?:node:)?(?:fs|net|http|https|child_process)['\"])"
)

_CPP_BANNED = re.compile(
    r"(?:\b(?:system|popen|_popen|fork|execve|execl|execvp|WinExec|ShellExecute|CreateProcess|std::exit|_Exit|quick_exit)\s*\("
    r"|#\s*include\s*[<\"](?:windows\.h|winsock2\.h|sys/socket\.h|unistd\.h|filesystem|fstream)[>\"])"
)
_CPP_ASSERT_LINE = re.compile(r"^\s*assert\s*\((.*)\)\s*;\s*$")
_SQL_BANNED = re.compile(r"\b(?:ATTACH\s+DATABASE|LOAD_EXTENSION|PRAGMA\s+writable_schema)\b", re.IGNORECASE)


def strip_think(text: str) -> str:
    return THINK_BLOCK.sub("", text).strip()


def extract_code(text: str) -> str:
    cleaned = strip_think(text)
    matches = CODE_BLOCK.findall(cleaned)
    return matches[-1].rstrip() if matches else cleaned


def cache_stats() -> dict:
    with _CACHE_LOCK:
        return {**_CACHE_STATS, "size": len(_VERIFY_CACHE)}


def check_ast_safety(code: str, filename: str = "<solution>") -> str | None:
    tree = ast.parse(code, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BANNED_MODULES:
                    return f"banned_import:{root}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in _BANNED_MODULES:
                return f"banned_import:{root}"
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRS:
                return f"banned_attr:{node.attr}"
            if isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr in {"exit", "modules", "_getframe"}:
                return f"banned_attr:sys.{node.attr}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_CALLS:
                return f"banned_call:{node.func.id}"
    return None


def check_js_safety(code: str) -> str | None:
    m = _JS_TS_BANNED.search(code)
    return f"banned_js:{m.group(0)}" if m else None


def check_cpp_safety(code: str) -> str | None:
    m = _CPP_BANNED.search(code)
    return f"banned_cpp:{m.group(0).strip()}" if m else None


def check_sql_safety(code: str) -> str | None:
    m = _SQL_BANNED.search(code)
    return f"banned_sql:{m.group(0)}" if m else None


def _fuzzy_replace_once(src: str, search_str: str, replace_str: str) -> tuple[str | None, str]:
    exact = src.count(search_str)
    if exact == 1:
        return src.replace(search_str, replace_str, 1), "ok"
    if exact > 1:
        return None, "ambiguous_search"

    src_lines = src.splitlines(keepends=True)
    needle_lines = [l.rstrip() for l in search_str.splitlines()]
    if not needle_lines:
        return None, "empty_search"

    hits = []
    m = len(needle_lines)
    for i in range(len(src_lines) - m + 1):
        if [l.rstrip() for l in src_lines[i : i + m]] == needle_lines:
            hits.append(i)
    if len(hits) == 1:
        idx = hits[0]
        tail_nl = "\n" if src_lines[idx + m - 1].endswith("\n") and not replace_str.endswith("\n") else ""
        rebuilt = "".join(src_lines[:idx]) + replace_str + tail_nl + "".join(src_lines[idx + m :])
        return rebuilt, "ok_fuzzy"
    return None, ("ambiguous_search" if len(hits) > 1 else "search_not_found")


def apply_search_replace_blocks(files: dict[str, str], response_text: str) -> tuple[dict[str, str] | None, str]:
    # Scan the whole reply: models often put each file's patch in its own fenced block with the
    # `### FILE:` header outside it, so narrowing to the last fence would silently drop files.
    cleaned = strip_think(response_text)
    updated = dict(files)
    file_sections = FILE_PATCH_BLOCK.findall(cleaned)
    if not file_sections:
        if len(updated) == 1 and PATCH_BLOCK.search(cleaned):
            file_sections = [(next(iter(updated)), cleaned)]
        else:
            return None, "no_patch_blocks"

    applied = 0
    for raw_path, body in file_sections:
        rel = os.path.normpath(raw_path.strip().strip("`'\"")).replace("\\", "/")
        if rel.startswith("/") or ".." in rel.split("/"):
            return None, f"unsafe_path:{rel}"
        if rel not in updated:
            return None, f"unknown_file:{rel}"
        patches = PATCH_BLOCK.findall(body)
        if not patches:
            return None, f"empty_patch_for:{rel}"
        src = updated[rel]
        for search_str, replace_str in patches:
            new_src, status = _fuzzy_replace_once(src, search_str, replace_str)
            if new_src is None:
                return None, f"{status}:{rel}"
            src = new_src
            applied += 1
        updated[rel] = src

    return (updated, "ok") if applied > 0 else (None, "no_edits_applied")


def _build_py_harness(test_code: str, marker: str, nonce: str) -> tuple[str, int]:
    lines = [
        "import os as __ss_os, sys as __ss_sys, time as __ss_time; __ss_sys.path.insert(0, __ss_os.getcwd())",
        "__ss_passed, __ss_total, __ss_err, __ss_algo_ms = 0, 0, 'ok', 0.0",
    ]
    try:
        tree = ast.parse(test_code)
        nodes = tree.body
    except SyntaxError:
        nodes = []

    assert_count = 0
    if nodes:
        for node in nodes:
            seg = ast.get_source_segment(test_code, node) or ast.unparse(node)
            if isinstance(node, ast.Assert):
                assert_count += 1
                lines.append("__ss_total += 1; __ss_t0 = __ss_time.perf_counter()")
                lines.append("try:")
                for al in seg.splitlines():
                    lines.append(f"    {al}")
                lines.append("    __ss_passed += 1")
                lines.append("except BaseException as __ss_e:")
                lines.append("    if __ss_err == 'ok': __ss_err = f'{type(__ss_e).__name__}: {__ss_e}'")
                lines.append("finally:")
                lines.append("    __ss_algo_ms += (__ss_time.perf_counter() - __ss_t0) * 1000.0")
            else:
                lines.append(seg)
    if assert_count == 0:
        assert_count = 1
        lines.append("__ss_total = 1; __ss_t0 = __ss_time.perf_counter()\ntry:")
        for tl in test_code.splitlines():
            lines.append(f"    {tl}")
        lines.append("    __ss_passed = 1\nexcept BaseException as __ss_e:\n    __ss_err = f'{type(__ss_e).__name__}: {__ss_e}'")
        lines.append("finally:\n    __ss_algo_ms += (__ss_time.perf_counter() - __ss_t0) * 1000.0")

    lines.append(
        f"import json as __ss_json; print({(marker + nonce + ':')!r} + __ss_json.dumps([__ss_passed, __ss_total, __ss_err, round(__ss_algo_ms, 3)]))"
    )
    return "\n".join(lines), assert_count


def _build_node_harness(test_code: str, marker: str, nonce: str, is_ts: bool = False) -> tuple[str, int]:
    top_imports, body_lines = [], []
    assert_count = 0
    for line in test_code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("import ", "export ")) and is_ts:
            top_imports.append(line)
        elif stripped.startswith(("assert(", "assert.")):
            assert_count += 1
            body_lines.append("__ss_t += 1; __ss_t0 = performance.now();")
            body_lines.append(
                f"try {{ {stripped}\n __ss_p += 1; }} catch (e: any) {{ if (__ss_err === 'ok') __ss_err = (e && e.message) ? e.name + ': ' + e.message : String(e); }} finally {{ __ss_ms += performance.now() - __ss_t0; }}"
                if is_ts
                else f"try {{ {stripped}\n __ss_p += 1; }} catch (e) {{ if (__ss_err === 'ok') __ss_err = (e && e.message) ? e.name + ': ' + e.message : String(e); }} finally {{ __ss_ms += performance.now() - __ss_t0; }}"
            )
        else:
            body_lines.append(line)

    if assert_count == 0:
        assert_count = 1
        body_lines.append("__ss_t = 1; __ss_p = 1;")

    header = (
        ["import assert from 'node:assert/strict';", *top_imports]
        if is_ts
        else ["const assert = require('node:assert/strict');"]
    )
    body = [
        *header,
        "process.exit = (() => { throw new Error('banned:process.exit'); }) as any;" if is_ts else "process.exit = () => { throw new Error('banned:process.exit'); };",
        "(globalThis as any).fetch = () => { throw new Error('banned:fetch'); };" if is_ts else "globalThis.fetch = () => { throw new Error('banned:fetch'); };",
        "await (async () => {" if is_ts else "(async () => {",
        "let __ss_p = 0, __ss_t = 0, __ss_err = 'ok', __ss_ms = 0, __ss_t0 = 0;",
        *body_lines,
        f"console.log({json.dumps(marker + nonce + ':')} + JSON.stringify([__ss_p, __ss_t, __ss_err, +__ss_ms.toFixed(3)]));",
        "})();",
    ]
    return "\n".join(body), assert_count


def _build_cpp_harness(test_code: str, marker: str, nonce: str) -> tuple[str, list[str], int]:
    includes, main_lines = [], [
        "int main() {",
        "    int __ss_p = 0, __ss_t = 0;",
        "    std::string __ss_err = \"ok\";",
    ]
    assert_count = 0
    for line in test_code.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#include"):
            includes.append(line)
            continue
        m = _CPP_ASSERT_LINE.match(line)
        if m:
            assert_count += 1
            ex = m.group(1).strip()
            escaped = ex.replace("\\", "\\\\").replace('"', '\\"')
            main_lines.append("    __ss_t += 1;")
            main_lines.append(
                f"    try {{ if (static_cast<bool>({ex})) {{ __ss_p += 1; }} else if (__ss_err == \"ok\") {{ __ss_err = \"AssertFailed#{assert_count}: {escaped}\"; }} }} "
                f"catch (const std::exception& e) {{ if (__ss_err == \"ok\") __ss_err = std::string(\"Exception: \") + e.what(); }} "
                f"catch (...) {{ if (__ss_err == \"ok\") __ss_err = \"UnknownException#{assert_count}\"; }}"
            )
        else:
            main_lines.append(f"    {line}")

    if assert_count == 0:
        assert_count = 1
        main_lines.append("    __ss_t = 1; __ss_p = 1;")

    std_headers = [
        "#include <iostream>", "#include <vector>", "#include <string>", "#include <algorithm>",
        "#include <unordered_map>", "#include <unordered_set>", "#include <map>", "#include <set>",
        "#include <queue>", "#include <stack>", "#include <numeric>", "#include <cmath>",
        "#include <cstdint>", "#include <optional>", "#include <utility>", "#include <functional>", "#include <chrono>",
        *includes,
        # JSON-escape the error text: failing asserts like `s == "abc"` or exception messages contain quotes.
        r"""static std::string __ss_esc(const std::string& s) { std::string o; for (unsigned char c : s) { if (c == '"' || c == '\\') { o += '\\'; o += static_cast<char>(c); } else if (c < 0x20) { o += ' '; } else { o += static_cast<char>(c); } } return o; }""",
    ]
    main_lines.insert(3, "    auto __ss_t0 = std::chrono::steady_clock::now();")
    main_lines.append(
        "    double __ss_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - __ss_t0).count();"
    )
    main_lines.append(
        f'    std::cout << "{marker}{nonce}:[" << __ss_p << "," << __ss_t << ",\\"" << __ss_esc(__ss_err) << "\\"," << __ss_ms << "]\\n";'
    )
    main_lines.append("    return 0;\n}")
    return "\n".join(std_headers), main_lines, assert_count


def _verify_sql(solution_text: str, test_code: str, starter_code: str, files: dict[str, str] | None) -> tuple[float, dict]:
    """In-memory SQLite3 execution verifier (zero subprocess overhead, <1ms, supports single queries & schema migrations!).
    `starter_code` holds initial DDL/seed SQL; `test_code` is Python assertions against `conn` and `rows`."""
    t0 = time.perf_counter()
    if files:
        patched, p_status = apply_search_replace_blocks(files, solution_text)
        if patched is None:
            partial = 0.05 if p_status.startswith(("search_not_found", "ambiguous_search")) else 0.0
            return partial, {"status": p_status, "passed": 0, "total": 1, "ms": 0.0}
        sql_script = "\n\n".join(patched[k] for k in sorted(patched) if k.endswith(".sql"))
    else:
        sql_script = extract_code(solution_text)

    if not sql_script:
        return -0.5, {"status": "empty", "passed": 0, "total": 1, "ms": 0.0}
    v = check_sql_safety(sql_script)
    if v:
        return -0.5, {"status": v, "passed": 0, "total": 1, "ms": 0.0}

    conn = sqlite3.connect(":memory:")
    conn.set_authorizer(
        lambda action, *_a: sqlite3.SQLITE_DENY if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_READ and 0) else sqlite3.SQLITE_OK
    )
    try:
        if starter_code:
            conn.executescript(starter_code)
        # If single SELECT/WITH query (and not multi-statement DDL), execute directly to capture result `rows`
        stripped_sql = sql_script.strip().rstrip(";")
        if re.match(r"^(?:SELECT|WITH)\b", stripped_sql, re.IGNORECASE) and ";" not in stripped_sql:
            cur = conn.execute(stripped_sql)
            rows = cur.fetchall()
        else:
            conn.executescript(sql_script)
            rows = []
    except sqlite3.Error as e:
        conn.close()
        return 0.0, {"status": f"sql_error: {e}", "passed": 0, "total": 1, "ms": round((time.perf_counter() - t0) * 1000, 2)}

    # Evaluate top-level Python assertions against `conn` and `rows` for dense partial credit!
    passed, first_err = 0, "ok"
    ns = {"conn": conn, "rows": rows}
    try:
        try:
            tree = ast.parse(test_code)
        except SyntaxError as e:
            return 0.0, {"status": f"test_syntax_error: {e.msg}", "passed": 0, "total": 1, "ms": round((time.perf_counter() - t0) * 1000, 2)}
        # Count every assert up front so a setup failure leaves the remaining asserts as failed, not uncounted.
        total = sum(isinstance(node, ast.Assert) for node in tree.body)
        for node in tree.body:
            code_obj = compile(ast.Module(body=[node], type_ignores=[]), "<sql_test>", "exec")
            try:
                exec(code_obj, ns)
                if isinstance(node, ast.Assert):
                    passed += 1
            except Exception as e:
                if first_err == "ok":
                    first_err = f"{type(e).__name__}: {e}"
                # A failed setup line (e.g. querying a column the solution never created) means later asserts
                # depend on missing state — stop here instead of crashing the whole GRPO run.
                if not isinstance(node, ast.Assert):
                    break
    finally:
        conn.close()

    total = total or 1
    ms = round((time.perf_counter() - t0) * 1000, 2)
    reward = 1.0 if passed == total else round(0.1 + 0.9 * (passed / total), 4)
    return reward, {"status": first_err, "passed": passed, "total": total, "ms": ms}


def _safe_write_tree(root: str, files: dict[str, str]) -> str | None:
    for rel, content in files.items():
        norm = os.path.normpath(rel)
        full = os.path.abspath(os.path.join(root, norm))
        if not full.startswith(os.path.abspath(root) + os.sep):
            return f"path_escape:{rel}"
        if norm.endswith(".py"):
            try:
                v = check_ast_safety(content, filename=rel)
            except SyntaxError as e:
                return f"syntax_error:{rel}:{e.msg}"
            if v:
                return f"{v}@{rel}"
        elif norm.endswith((".js", ".mjs", ".cjs", ".ts", ".mts", ".cts")):
            v = check_js_safety(content)
            if v:
                return f"{v}@{rel}"
        elif norm.endswith((".cpp", ".cc", ".cxx", ".hpp", ".h")):
            v = check_cpp_safety(content)
            if v:
                return f"{v}@{rel}"
        elif norm.endswith(".sql"):
            v = check_sql_safety(content)
            if v:
                return f"{v}@{rel}"
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
    return None


def verify(
    solution_text: str,
    test_code: str,
    starter_code: str = "",
    files: dict[str, str] | None = None,
    lang: str = "python",
    timeout: float = 4.0,
    perf_budget_ms: float = 0.0,
    use_cache: bool = True,
) -> tuple[float, dict]:
    """5-Language (Python + JS + TypeScript + C++23 + SQL) verifier with SHA-256 memoization & optional complexity speed budget."""
    if not solution_text or not strip_think(solution_text):
        return -0.5, {"status": "empty", "passed": 0, "total": 0, "ms": 0.0}

    lang = (lang or "python").lower()
    cache_key = ""
    if use_cache:
        raw_key = json.dumps([lang, strip_think(solution_text), test_code, starter_code, files, perf_budget_ms], sort_keys=True)
        cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        with _CACHE_LOCK:
            if cache_key in _VERIFY_CACHE:
                _VERIFY_CACHE.move_to_end(cache_key)
                _CACHE_STATS["hits"] += 1
                sc, inf = _VERIFY_CACHE[cache_key]
                return sc, {**inf, "cached": True}
            _CACHE_STATS["misses"] += 1

    score, info = _verify_uncached(solution_text, test_code, starter_code, files, lang, timeout)
    if score == 1.0 and perf_budget_ms > 0.0:
        # Measure pure algorithmic execution inside the harness (`algo_ms`) so subprocess startup jitter never penalizes!
        exec_ms = info.get("algo_ms", info.get("ms", 0.0))
        if exec_ms > perf_budget_ms:
            ratio = min(1.0, perf_budget_ms / max(exec_ms, 0.01))
            score = round(0.75 + 0.20 * ratio, 4)  # Capped at 0.95 (< 1.0) so RLVR strongly prefers O(n)/O(n log n)!
            info = {**info, "status": f"perf_budget_exceeded:{exec_ms:.1f}ms>{perf_budget_ms:.1f}ms"}

    if use_cache and info.get("status") != "timeout":
        with _CACHE_LOCK:
            _VERIFY_CACHE[cache_key] = (score, dict(info))
            if len(_VERIFY_CACHE) > _CACHE_MAX:
                _VERIFY_CACHE.popitem(last=False)
    return score, info


def _verify_uncached(
    solution_text: str,
    test_code: str,
    starter_code: str,
    files: dict[str, str] | None,
    lang: str,
    timeout: float,
) -> tuple[float, dict]:
    if lang in ("sql", "sqlite"):
        return _verify_sql(solution_text, test_code, starter_code, files)

    is_cpp = lang in ("cpp", "c++", "cxx", "cc")
    is_ts = lang in ("typescript", "ts")
    is_js = lang in ("javascript", "js", "node")

    if (is_js or is_ts) and not _NODE_BIN:
        return 0.0, {"status": "node_not_installed", "passed": 0, "total": 0, "ms": 0.0}
    if is_cpp and not _GPP_BIN:
        return 0.0, {"status": "gpp_not_installed", "passed": 0, "total": 0, "ms": 0.0}

    marker = "__NEUTRON_V9_"
    nonce = os.urandom(8).hex()

    # ponytail: OS temp dir + AST/JS/CPP/SQL checks + nonce secures single-tenant RLVR; add Docker/gVisor for multi-tenant SaaS.
    with tempfile.TemporaryDirectory(prefix="neutron_") as tmp:
        if is_cpp:
            headers_src, main_lines, expected_total = _build_cpp_harness(test_code, marker, nonce)
            main_src = "\n".join(main_lines)
            cpp_units = []
            if files:
                patched, p_status = apply_search_replace_blocks(files, solution_text)
                if patched is None:
                    partial = 0.05 if p_status.startswith(("search_not_found", "ambiguous_search")) else 0.0
                    return partial, {"status": p_status, "passed": 0, "total": expected_total, "ms": 0.0}
                err = _safe_write_tree(tmp, patched)
                if err:
                    return (-0.5 if err.startswith("banned_") else 0.0), {"status": err, "passed": 0, "total": expected_total, "ms": 0.0}
                cpp_units = [os.path.join(tmp, os.path.normpath(p)) for p in patched if p.endswith((".cpp", ".cc", ".cxx"))]
                entry = os.path.join(tmp, "__ss_main.cpp")
                with open(entry, "w", encoding="utf-8") as f:
                    f.write(f"{headers_src}\n\n{main_src}\n")
                cpp_units.append(entry)
            else:
                code = extract_code(solution_text)
                if not code:
                    return -0.5, {"status": "empty", "passed": 0, "total": expected_total, "ms": 0.0}
                v = check_cpp_safety(code)
                if v:
                    return -0.5, {"status": v, "passed": 0, "total": expected_total, "ms": 0.0}
                full_code = f"{starter_code}\n\n{code}".strip() if starter_code else code
                entry = os.path.join(tmp, "__ss_main.cpp")
                with open(entry, "w", encoding="utf-8") as f:
                    f.write(f"{headers_src}\n\n{full_code}\n\n{main_src}\n")
                cpp_units = [entry]

            exe_path = os.path.join(tmp, "runner.exe" if os.name == "nt" else "runner")
            compile_cmd = [_GPP_BIN, "-std=c++23", "-O2", f"-I{tmp}", *cpp_units, "-o", exe_path]
            t0 = time.perf_counter()
            try:
                cproc = subprocess.run(compile_cmd, cwd=tmp, env=_SAFE_ENV, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=max(15.0, timeout * 3))
            except subprocess.TimeoutExpired:
                return 0.0, {"status": "cpp_compile_timeout", "passed": 0, "total": expected_total, "ms": 0.0}
            if cproc.returncode != 0:
                first_err = next((l.strip() for l in cproc.stderr.splitlines() if "error:" in l), "cpp_compile_error")
                return 0.0, {"status": f"compile_error: {first_err[:140]}", "passed": 0, "total": expected_total, "ms": 0.0}
            run_cmd = [exe_path]
        else:
            harness_src, expected_total = (
                _build_node_harness(test_code, marker, nonce, is_ts=is_ts)
                if (is_js or is_ts)
                else _build_py_harness(test_code, marker, nonce)
            )
            ext = ".mts" if is_ts else (".cjs" if is_js else ".py")
            if files:
                patched, p_status = apply_search_replace_blocks(files, solution_text)
                if patched is None:
                    partial = 0.05 if p_status.startswith(("search_not_found", "ambiguous_search")) else 0.0
                    return partial, {"status": p_status, "passed": 0, "total": expected_total, "ms": 0.0}
                err = _safe_write_tree(tmp, patched)
                if err:
                    return (-0.5 if err.startswith("banned_") else 0.0), {"status": err, "passed": 0, "total": expected_total, "ms": 0.0}
                entry = os.path.join(tmp, f"__ss_check{ext}")
                with open(entry, "w", encoding="utf-8") as f:
                    f.write(harness_src)
            else:
                code = extract_code(solution_text)
                if not code:
                    return -0.5, {"status": "empty", "passed": 0, "total": expected_total, "ms": 0.0}
                if is_js or is_ts:
                    v = check_js_safety(code)
                    if v:
                        return -0.5, {"status": v, "passed": 0, "total": expected_total, "ms": 0.0}
                else:
                    try:
                        v = check_ast_safety(code)
                    except SyntaxError as e:
                        return 0.0, {"status": f"syntax_error: {e.msg}", "passed": 0, "total": expected_total, "ms": 0.0}
                    if v:
                        return -0.5, {"status": v, "passed": 0, "total": expected_total, "ms": 0.0}
                full_code = f"{starter_code}\n\n{code}".strip() if starter_code else code
                entry = os.path.join(tmp, f"__ss_check{ext}")
                with open(entry, "w", encoding="utf-8") as f:
                    f.write(f"{full_code}\n\n{harness_src}\n")

            if is_ts:
                run_cmd = [_NODE_BIN, *_ts_node_flags(), entry]
            elif is_js:
                run_cmd = [_NODE_BIN, entry]
            else:
                run_cmd = [sys.executable, "-I", "-B", entry]
            t0 = time.perf_counter()

        env = dict(_SAFE_ENV, PYTHONPATH=tmp)
        try:
            # errors="replace": a solution printing non-UTF-8 bytes must not raise UnicodeDecodeError here.
            proc = subprocess.run(run_cmd, cwd=tmp, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        except subprocess.TimeoutExpired:
            return 0.0, {"status": "timeout", "passed": 0, "total": expected_total, "ms": round(timeout * 1000, 1)}

    prefix = f"{marker}{nonce}:"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(prefix):
            payload = json.loads(line[len(prefix):])
            passed, total, err = payload[0], payload[1], payload[2]
            algo_ms = payload[3] if len(payload) > 3 else elapsed_ms
            frac = (passed / total) if total else 0.0
            reward = round(0.1 + 0.9 * frac, 4) if passed < total else 1.0
            return reward, {"status": err, "passed": passed, "total": total, "ms": elapsed_ms, "algo_ms": algo_ms}

    stderr_txt = proc.stderr or proc.stdout
    if (is_js or is_ts) and "SyntaxError" in stderr_txt:
        return 0.0, {"status": "ts_js_syntax_error", "passed": 0, "total": expected_total, "ms": elapsed_ms}
    err_lines = stderr_txt.strip().splitlines()
    msg = err_lines[-1] if err_lines else f"exit_{proc.returncode}"
    return 0.1, {"status": msg, "passed": 0, "total": expected_total, "ms": elapsed_ms}


def verify_batch(
    solutions: list[str],
    tests: list[str],
    starters: list[str] | None = None,
    files_list: list[dict[str, str] | None] | None = None,
    langs: list[str] | None = None,
    perf_budgets: list[float] | None = None,
    timeout: float = 4.0,
    max_workers: int | None = None,
) -> list[tuple[float, dict]]:
    n = len(solutions)
    starters = starters or [""] * n
    files_list = files_list or [None] * n
    langs = langs or ["python"] * n
    perf_budgets = perf_budgets or [0.0] * n
    workers = max_workers or min(n or 1, os.cpu_count() or 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(verify, s, t, st, fl, lg, timeout, pb)
            for s, t, st, fl, lg, pb in zip(solutions, tests, starters, files_list, langs, perf_budgets)
        ]
        results = []
        for f in futs:
            try:
                results.append(f.result())
            except Exception as e:  # A verifier bug must never kill a paid training run; score it 0 and keep going.
                print(f"[sandbox] verifier crashed: {type(e).__name__}: {e}", flush=True)
                results.append((0.0, {"status": f"verifier_crash: {type(e).__name__}: {e}", "passed": 0, "total": 1, "ms": 0.0}))
        return results


def grpo_reward_func(completions, test, starter_code=None, files=None, lang=None, perf_budget_ms=None, **kwargs) -> list[float]:
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    n = len(texts)
    starters = list(starter_code) if starter_code is not None else [""] * n
    langs = list(lang) if lang is not None else ["python"] * n
    budgets = [float(b or 0.0) for b in (list(perf_budget_ms) if perf_budget_ms is not None else [0.0] * n)]
    parsed_files = [
        (json.loads(f) if isinstance(f, str) and f else f)
        for f in (list(files) if files is not None else [None] * n)
    ]
    return [score for score, _ in verify_batch(texts, list(test), starters, parsed_files, langs, budgets)]


def reasoning_format_reward_func(completions, files=None, **kwargs) -> list[float]:
    n = len(completions)
    raw_files = list(files) if files is not None else [None] * n
    out = []
    for comp, fl in zip(completions, raw_files):
        text = (comp[0]["content"] if isinstance(comp, list) else comp).strip()
        is_repo = bool(json.loads(fl) if isinstance(fl, str) and fl else fl)
        score = 0.05 if THINK_BLOCK.search(text) else 0.0
        body = strip_think(text)
        if is_repo:
            if "<<<<<<< SEARCH" in body and ">>>>>>> REPLACE" in body:
                score += 0.1
        elif body.startswith("```") and body.endswith("```"):
            score += 0.1
        if len(text) > 6000:
            score -= min(0.1, (len(text) - 6000) / 20000)
        out.append(round(score, 4))
    return out


if __name__ == "__main__":
    # 1. Python + SHA-256 memoization check
    r1, i1 = verify("<think>t</think>\n```python\ndef f(x): return 4\n```", "assert f(2) == 4\nassert f(3) == 6")
    r2, i2 = verify("<think>different think</think>\n```python\ndef f(x): return 4\n```", "assert f(2) == 4\nassert f(3) == 6")
    assert r1 == 0.55 and r2 == 0.55 and i2.get("cached") is True, (r1, i1, i2)

    # 2. Native TypeScript + C++23 checks
    assert verify("```ts\nconst sq = (n: number): number => n * n;\n```", "assert.equal(sq(8), 64);", lang="typescript")[0] == 1.0
    assert verify("```cpp\nint dbl(int x) { return x * 2; }\n```", "assert(dbl(21) == 42);", lang="cpp")[0] == 1.0

    # 3. In-memory SQLite SQL window-function query + anti-cheat (`ATTACH DATABASE` blocked)
    seed = "CREATE TABLE sales(dept TEXT, emp TEXT, rev INT); INSERT INTO sales VALUES ('eng','a',100),('eng','b',250),('ops','c',180);"
    sql_q = "<think>window</think>\n```sql\nSELECT dept, emp, rev FROM (SELECT dept, emp, rev, RANK() OVER (PARTITION BY dept ORDER BY rev DESC) rk FROM sales) WHERE rk = 1 ORDER BY dept;\n```"
    r, info = verify(sql_q, "assert rows == [('eng', 'b', 250), ('ops', 'c', 180)]", starter_code=seed, lang="sql")
    assert r == 1.0 and info["passed"] == 1, (r, info)

    r_bad, info_bad = verify("```sql\nATTACH DATABASE ':memory:' AS aux;\n```", "assert True", lang="sql")
    assert r_bad == -0.5 and "banned_sql" in info_bad["status"], (r_bad, info_bad)

    # 4. In-harness algorithmic complexity / performance budget (`O(n)` gets 1.0 vs `O(n^2)` penalized <1.0!)
    perf_test = "arr = list(range(4000)) + [100]\nassert has_dup(arr) is True\nassert has_dup(list(range(4000))) is False"
    r_fast, i_fast = verify("```python\ndef has_dup(a): return len(set(a)) < len(a)\n```", perf_test, perf_budget_ms=50.0)
    r_slow, i_slow = verify(
        "```python\ndef has_dup(a):\n    for i in range(len(a)):\n        for j in range(i + 1, len(a)):\n            if a[i] == a[j]: return True\n    return False\n```",
        perf_test,
        perf_budget_ms=50.0,
    )
    assert r_fast == 1.0 and 0.70 <= r_slow < 1.0 and "perf_budget_exceeded" in i_slow["status"], (r_fast, r_slow, i_slow)

    print(f"sandbox.py (NeutronAI v0.9) self-check passed (5-lang Py/JS/TS/C++23/SQL + algo_ms speed gate + SHA-256 cache={cache_stats()}).")
