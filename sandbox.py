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
    "importlib", "inspect", "gc", "marshal", "pickle", "atexit", "__main__",
})
_BANNED_ATTRS = frozenset({
    "system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve",
    "spawnvp", "spawnvpe", "execl", "execle", "execlp", "execlpe", "execv", "execve",
    "execvp", "execvpe", "_exit", "kill", "unlink", "rmdir", "makedirs",
    "__subclasses__", "__globals__", "__builtins__", "__code__", "__closure__", "f_globals", "f_locals",
    "__module__", "__getattribute__", "settrace", "setprofile",
})
# Banned only on an imported module (`os.remove`) or via `from os import remove`; `lst.remove(x)` stays legal.
_BANNED_MODULE_ATTRS = frozenset({"remove", "removedirs", "rename", "replace", "truncate", "__dict__"})
_BANNED_CALLS = frozenset({
    "exit", "quit", "eval", "exec", "compile", "__import__", "open", "input",
    "breakpoint", "getattr", "setattr", "delattr", "globals", "locals", "vars",
})
_BANNED_NAMES = frozenset({"__builtins__", "__module__"})
# Single-file Python solutions share a namespace with the test asserts, so rebinding one of these at module
# level (e.g. `def abs(x): return 0`) would change what the asserts compute.
_PY_GUARDED_NAMES = frozenset({
    "abs", "all", "any", "bool", "callable", "dict", "enumerate", "filter", "float", "frozenset", "getattr",
    "hasattr", "hash", "id", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "object", "print", "range", "repr", "reversed", "round", "set", "sorted", "str", "sum", "tuple",
    "type", "zip", "AssertionError", "BaseException", "Exception", "KeyError", "IndexError", "TypeError",
    "ValueError", "ZeroDivisionError", "StopIteration",
})
_PY_GUARDED_MODULES = frozenset({
    "math", "cmath", "operator", "collections", "itertools", "functools", "heapq", "bisect", "re", "json",
    "string", "decimal", "fractions", "statistics", "random", "sys", "os", "time", "copy",
})

_JS_TS_BANNED = re.compile(
    r"\b(?:child_process|process\.exit|process\.kill|process\.binding|process\.dlopen|__ss|"
    r"require\s*\(\s*['\"](?:node:)?(?:fs|net|http|https|dgram|cluster|worker_threads|vm|inspector|os|dns|tls)['\"]|"
    r"from\s+['\"](?:node:)?(?:fs|net|http|https|child_process|dgram|cluster|worker_threads|vm|inspector|os|dns|tls)['\"]|"
    r"import\s*\(\s*['\"](?:node:)?(?:fs|net|http|https|child_process)['\"])"
)

_CPP_BANNED = re.compile(
    r"(?:\b(?:system|popen|_popen|fork|execve|execl|execvp|WinExec|ShellExecute|CreateProcess|std::exit|_Exit|quick_exit)\s*\("
    r"|#\s*include\s*[<\"](?:windows\.h|winsock2\.h|sys/socket\.h|sys/syscall\.h|unistd\.h|fcntl\.h|dlfcn\.h|filesystem|fstream)[>\"]"
    # The harness reads its report prefix from stdin before main(); solutions may not touch stdin, files,
    # harness identifiers (`__ss`), or token pasting (`##`, which could rebuild any banned name).
    r"|\b__ss|##|\b(?:cin|wcin|stdin)\b|/dev/|/proc/"
    r"|\b(?:getchar|scanf|fscanf|vscanf|gets|fgets|fgetc|getc|fread|fopen|freopen|fdopen|syscall)\s*\("
    r"|(?:::|\b)read\s*\(\s*(?:0|STDIN_FILENO)\b)"
)
_CPP_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)", re.MULTILINE)
_CPP_ASSERT_LINE = re.compile(r"^\s*assert\s*\((.*)\)\s*;\s*$")
_SQL_BANNED = re.compile(r"\b(?:ATTACH\s+DATABASE|LOAD_EXTENSION|PRAGMA\s+writable_schema)\b", re.IGNORECASE)


def strip_think(text: str) -> str:
    text = THINK_BLOCK.sub("", text)
    # Thinking models (Qwen3.5) often get `<think>` opened by the chat template inside the prompt, so the
    # completion only contains `...reasoning...</think> answer`. Keep just the answer.
    if "</think>" in text and "<think>" not in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def extract_code(text: str) -> str:
    cleaned = strip_think(text)
    matches = CODE_BLOCK.findall(cleaned)
    return matches[-1].rstrip() if matches else cleaned


def cache_stats() -> dict:
    with _CACHE_LOCK:
        return {**_CACHE_STATS, "size": len(_VERIFY_CACHE)}


def _identifiers(node: ast.AST):
    """Every identifier a node introduces or references (names, attrs, params, defs, import aliases, ...)."""
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, ast.Attribute):
        yield node.attr
    elif isinstance(node, ast.arg):
        yield node.arg
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node.name
    elif isinstance(node, ast.alias):
        yield node.name
        if node.asname:
            yield node.asname
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        yield from node.names
    elif isinstance(node, ast.keyword) and node.arg:
        yield node.arg
    elif isinstance(node, ast.ExceptHandler) and node.name:
        yield node.name


def check_ast_safety(code: str, filename: str = "<solution>") -> str | None:
    tree = ast.parse(code, filename=filename)
    imported = {
        (a.asname or a.name).split(".")[0]
        for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in n.names
    }
    for node in ast.walk(tree):
        for ident in _identifiers(node):
            if ident.startswith("__ss"):  # harness internals
                return f"banned_name:{ident}"
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            return f"banned_name:{node.id}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BANNED_MODULES:
                    return f"banned_import:{root}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in _BANNED_MODULES:
                return f"banned_import:{root}"
            for alias in node.names:
                fs_module = root in ("os", "posix", "nt", "pathlib", "io")  # `from dataclasses import replace` is fine
                if alias.name in _BANNED_ATTRS or (fs_module and alias.name in _BANNED_MODULE_ATTRS):
                    return f"banned_import:{node.module}.{alias.name}"
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRS:
                return f"banned_attr:{node.attr}"
            if isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr in {"exit", "modules", "_getframe"}:
                return f"banned_attr:sys.{node.attr}"
            if node.attr in _BANNED_MODULE_ATTRS and isinstance(node.value, ast.Name) and node.value.id in imported:
                return f"banned_attr:{node.value.id}.{node.attr}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_CALLS:
                return f"banned_call:{node.func.id}"
            if node.func.id == "type" and len(node.args) == 3:  # dynamic class creation can spoof __module__
                return "banned_call:type/3"
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for tgt in targets:
                for t in ast.walk(tgt):
                    if not (isinstance(t, ast.Attribute) and isinstance(t.ctx, (ast.Store, ast.Del))):
                        continue
                    if t.attr.startswith("__") and t.attr.endswith("__"):  # e.g. Counter.__eq__ = ...
                        return f"banned_dunder_assign:{t.attr}"
                    if isinstance(t.value, ast.Name) and t.value.id in imported:  # e.g. math.isclose = ...
                        return f"banned_module_patch:{t.value.id}.{t.attr}"
    return None


def _module_level_bindings(stmts: list[ast.stmt]):
    """(name, how) for names bound at module scope, descending into if/for/while/with/try but not defs."""
    for s in stmts:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield s.name, "def"
            continue
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            for a in s.names:
                if a.name != "*":
                    real = a.name.split(".")[0] if isinstance(s, ast.Import) else None
                    bound = (a.asname or a.name).split(".")[0]
                    yield bound, ("import_same" if real == bound else "import")
            continue
        targets = []
        if isinstance(s, (ast.Assign,)):
            targets = s.targets
        elif isinstance(s, (ast.AnnAssign, ast.AugAssign)):
            targets = [s.target]
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            targets = [s.target]
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            targets = [i.optional_vars for i in s.items if i.optional_vars is not None]
        for tgt in targets:
            for t in ast.walk(tgt):
                if isinstance(t, ast.Name):
                    yield t.id, "assign"
        for sub in ("body", "orelse", "finalbody"):
            if hasattr(s, sub) and not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield from _module_level_bindings(getattr(s, sub))
        for h in getattr(s, "handlers", []):
            if h.name:
                yield h.name, "assign"
            yield from _module_level_bindings(h.body)


def check_py_shadowing(code: str) -> str | None:
    """Single-file mode: reject solutions that rebind builtins / stdlib modules the test asserts rely on."""
    tree = ast.parse(code)
    for name, how in _module_level_bindings(tree.body):
        if name in _PY_GUARDED_NAMES:
            return f"shadows_builtin:{name}"
        if name in _PY_GUARDED_MODULES and how != "import_same":
            return f"shadows_module:{name}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            for name in node.names:
                if name in _PY_GUARDED_NAMES or name in _PY_GUARDED_MODULES:
                    return f"shadows_builtin:{name}"
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


# ---------------------------------------------------------------------------------------------------------------
# Anti-reward-hacking design (all three runtimes):
#   * The per-run report prefix (marker + nonce) is sent on stdin and never written into any file, so a solution
#     that reads its own source cannot forge a result line. A prologue that runs BEFORE solution code consumes stdin
#     and seals the prefix inside a closure, together with the original stdout writer and timer.
#   * Harness identifiers (`__ss*`) are banned in solution code, so the solution cannot call or alter them.
#   * Python: builtins + common stdlib modules are snapshotted and restored before every test statement, solutions
#     may not rebind guarded names, and comparisons inside asserts reject "always-equal" style objects.
#   * JS/TS: node:assert is frozen before the solution runs.  C++: solution macros are #undef'd before the harness,
#     and `==`/`!=` asserts go through a comparator that std types can't be hijacked in and that rejects custom
#     types compared against plain values.
# ---------------------------------------------------------------------------------------------------------------

_PY_PROLOGUE = r'''
import sys as __ss_sys, os as __ss_os, builtins as __ss_bi, operator as __ss_op, json as __ss_json, time as __ss_time
import math as __ss_m1, cmath as __ss_m2, collections as __ss_m3, itertools as __ss_m4, functools as __ss_m5
import heapq as __ss_m6, bisect as __ss_m7, re as __ss_m8, string as __ss_m9
__ss_sys.path.insert(0, __ss_os.getcwd())
__ss_pc = __ss_time.perf_counter
__ss_report = (lambda w, fl, d, r, p: lambda v: (w(p + d([v[0], v[1], v[2], r(v[3], 3)]) + "\n"), fl()))(
    __ss_sys.stdout.write, __ss_sys.stdout.flush, __ss_json.dumps, round, __ss_sys.stdin.readline().strip())
__ss_snap = tuple((m.__dict__, dict(m.__dict__)) for m in (
    __ss_bi, __ss_op, __ss_m1, __ss_m2, __ss_m3, __ss_m4, __ss_m5, __ss_m6, __ss_m7, __ss_m8, __ss_m9))


def __ss_restore(_snap=__ss_snap, _items=dict.items, _get=dict.get):
    for live, saved in _snap:
        for k, v in _items(saved):
            if _get(live, k) is not v:
                live[k] = v


__ss_D = (
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__bool__", "__len__", "__contains__", "__iter__",
    "__getitem__", "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
    "__floordiv__", "__rfloordiv__", "__mod__", "__rmod__", "__pow__", "__rpow__", "__neg__", "__pos__", "__abs__",
    "__int__", "__float__", "__index__", "__round__", "__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__",
)
__ss_OK = frozenset({
    "builtins", "collections", "_collections", "_collections_abc", "decimal", "_decimal", "_pydecimal", "fractions",
    "datetime", "_datetime", "_pydatetime", "numpy", "enum", "array", "ipaddress", "pathlib", "uuid", "typing",
    "abc", "numbers", "types", "math",
})


def __ss_honest(v, _type=type, _issub=issubclass, _mro=type.__dict__["__mro__"].__get__,
                _tdict=type.__dict__["__dict__"].__get__, _tmod=type.__dict__["__module__"].__get__,
                _plain=frozenset({int, float, complex, bool, str, bytes, type(None)}), _D=__ss_D, _ok=__ss_OK,
                _cache={}, _str=str, _dict=dict, _list=list, _tuple=tuple, _set=set, _fset=frozenset):
    """True if `v` (recursively) only uses operator methods from builtin/stdlib types."""
    stack = [v]
    budget = 200000
    while stack:
        x = stack.pop()
        budget -= 1
        if budget < 0:
            return True
        t = _type(x)
        if t in _plain:
            continue
        ok = _cache.get(t)
        if ok is None:
            ok = True
            for name in _D:
                for c in _mro(t):
                    if name in _tdict(c):
                        mod = _tmod(c)
                        ok = _type(mod) is _str and mod.partition(".")[0] in _ok
                        break
                if not ok:
                    break
            _cache[t] = ok
        if not ok:
            return False
        if _issub(t, _dict):
            stack.extend(_dict.keys(x))
            stack.extend(_dict.values(x))
        elif _issub(t, _list):
            stack.extend(_list.__iter__(x))
        elif _issub(t, _tuple):
            stack.extend(_tuple.__iter__(x))
        elif _issub(t, _set):
            stack.extend(_set.__iter__(x))
        elif _issub(t, _fset):
            stack.extend(_fset.__iter__(x))
    return True


__ss_hon = [0.0]  # ms spent in honesty checks, excluded from the perf-budget timing


def __ss_pair(a, b, _h=__ss_honest, _pc=__ss_pc, _acc=__ss_hon, _AE=AssertionError):
    t0 = _pc()
    try:
        if _h(a) is not _h(b):
            raise _AE("dishonest comparison: an object with custom operator methods was compared to a plain value")
    finally:
        _acc[0] += (_pc() - t0) * 1000.0


__ss_CMP = {
    "Eq": __ss_op.eq, "NotEq": __ss_op.ne, "Lt": __ss_op.lt, "LtE": __ss_op.le, "Gt": __ss_op.gt, "GtE": __ss_op.ge,
    "In": lambda a, b, _c=__ss_op.contains: _c(b, a), "NotIn": lambda a, b, _c=__ss_op.contains: not _c(b, a),
}
__ss_BIN = {
    "Add": __ss_op.add, "Sub": __ss_op.sub, "Mult": __ss_op.mul, "Div": __ss_op.truediv, "FloorDiv": __ss_op.floordiv,
    "Mod": __ss_op.mod, "Pow": __ss_op.pow, "MatMult": __ss_op.matmul, "LShift": __ss_op.lshift,
    "RShift": __ss_op.rshift, "BitOr": __ss_op.or_, "BitXor": __ss_op.xor, "BitAnd": __ss_op.and_,
}


def __ss_cmp(op, a, b, _p=__ss_pair, _t=__ss_CMP):
    _p(a, b)
    return _t[op](a, b)


def __ss_bin(op, a, b, _h=__ss_honest, _t=__ss_BIN, _pc=__ss_pc, _acc=__ss_hon, _AE=AssertionError):
    r = _t[op](a, b)
    t0 = _pc()
    try:
        # A custom object combined with a plain value that collapses into a plain value (e.g. `A() - 3.0 -> 0.0`).
        if _h(a) is not _h(b) and _h(r):
            raise _AE("dishonest arithmetic: a custom object combined with a plain value produced a plain value")
    finally:
        _acc[0] += (_pc() - t0) * 1000.0
    return r
'''
_PY_FUTURE = re.compile(r"^from[ \t]+__future__[ \t]+import[^\n]*\n?", re.MULTILINE)


def _hoist_future(code: str) -> tuple[str, str]:
    """`from __future__` imports must stay first in the file, ahead of the harness prologue."""
    futures = "".join(m.group(0).rstrip("\n") + "\n" for m in _PY_FUTURE.finditer(code))
    return futures, _PY_FUTURE.sub("", code)


class _PyAssertRewriter(ast.NodeTransformer):
    """Route comparisons and arithmetic inside top-level test asserts through honesty-checking helpers."""

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) != 1 or isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            return node
        return ast.Call(ast.Name("__ss_cmp", ast.Load()), [ast.Constant(type(node.ops[0]).__name__), node.left, node.comparators[0]], [])

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        return ast.Call(ast.Name("__ss_bin", ast.Load()), [ast.Constant(type(node.op).__name__), node.left, node.right], [])


def _build_py_harness(test_code: str) -> tuple[str, int]:
    lines = ["__ss_passed, __ss_total, __ss_err, __ss_algo_ms = 0, 0, 'ok', 0.0"]
    try:
        nodes = ast.parse(test_code).body
    except SyntaxError:
        nodes = []

    assert_count = 0
    if any(isinstance(n, ast.Assert) for n in nodes):
        for node in nodes:
            lines.append("__ss_restore()")
            if isinstance(node, ast.Assert):
                assert_count += 1
                seg = ast.unparse(ast.fix_missing_locations(_PyAssertRewriter().visit(node)))
                lines.append("__ss_total += 1; __ss_t1 = None; __ss_h0 = __ss_hon[0]; __ss_t0 = __ss_pc()")
                lines.append("try:")
                lines.extend(f"    {al}" for al in seg.splitlines())
                lines.append("    __ss_t1 = __ss_pc()")
                lines.append("    __ss_passed += 1")
                lines.append("except BaseException as __ss_e:")
                lines.append("    if __ss_err == 'ok': __ss_err = f'{type(__ss_e).__name__}: {__ss_e}'")
                lines.append("finally:")
                lines.append("    __ss_algo_ms += ((__ss_pc() if __ss_t1 is None else __ss_t1) - __ss_t0) * 1000.0 - (__ss_hon[0] - __ss_h0)")
            else:
                lines.append(ast.get_source_segment(test_code, node) or ast.unparse(node))
    else:
        # No top-level asserts (or unparsable test): the whole test passing without raising is one check.
        assert_count = 1
        lines.append("__ss_restore()")
        lines.append("__ss_total = 1; __ss_t0 = __ss_pc()\ntry:")
        lines.extend(f"    {tl}" for tl in test_code.splitlines())
        lines.append("    __ss_passed = 1\nexcept BaseException as __ss_e:\n    __ss_err = f'{type(__ss_e).__name__}: {__ss_e}'")
        lines.append("finally:\n    __ss_algo_ms += (__ss_pc() - __ss_t0) * 1000.0")

    lines.append("__ss_report([__ss_passed, __ss_total, __ss_err, __ss_algo_ms])")
    return "\n".join(lines), assert_count


# Runs before solution code: seal the stdin report prefix + original stdout writer/timer in closures, freeze assert.
_NODE_PROLOGUE_CJS = r"""const __ss_report = ((w, p) => (s) => w(p + s + '\n'))(process.stdout.write.bind(process.stdout), require('node:fs').readFileSync(0, 'utf8').trim());
const __ss_now = performance.now.bind(performance), __ss_json = JSON.stringify;
Object.freeze(require('node:assert')); Object.freeze(require('node:assert/strict'));
"""
_NODE_PROLOGUE_ESM = r"""import { readFileSync as __ss_rfs } from 'node:fs';
import __ss_a0 from 'node:assert';
import __ss_a1 from 'node:assert/strict';
Object.freeze(__ss_a0); Object.freeze(__ss_a1);
const __ss_report = ((w, p) => (s) => w(p + s + '\n'))(process.stdout.write.bind(process.stdout), __ss_rfs(0, 'utf8').trim());
const __ss_now = performance.now.bind(performance), __ss_json = JSON.stringify;
"""


def _build_node_harness(test_code: str, is_ts: bool = False) -> tuple[str, str, int]:
    """Return (prologue, harness_body, assert_count). The prologue must precede the solution code."""
    top_imports, body_lines = [], []
    assert_count = 0
    catch = "catch (e: any)" if is_ts else "catch (e)"
    for line in test_code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("import ", "export ")) and is_ts:
            top_imports.append(line)
        elif stripped.startswith(("assert(", "assert.")):
            assert_count += 1
            body_lines.append("__ss_t += 1; __ss_t0 = __ss_now();")
            body_lines.append(
                f"try {{ {stripped}\n __ss_p += 1; }} {catch} {{ if (__ss_err === 'ok') __ss_err = (e && e.message) ? e.name + ': ' + e.message : String(e); }} finally {{ __ss_ms += __ss_now() - __ss_t0; }}"
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
        "__ss_report('[' + __ss_p + ',' + __ss_t + ',' + __ss_json('' + __ss_err) + ',' + __ss_ms + ']');",
        "})();",
    ]
    return (_NODE_PROLOGUE_ESM if is_ts else _NODE_PROLOGUE_CJS), "\n".join(body), assert_count


_CPP_STD_HEADERS = [
    "#include <iostream>", "#include <vector>", "#include <string>", "#include <string_view>", "#include <algorithm>",
    "#include <unordered_map>", "#include <unordered_set>", "#include <map>", "#include <set>", "#include <array>",
    "#include <deque>", "#include <list>", "#include <tuple>", "#include <type_traits>",
    "#include <queue>", "#include <stack>", "#include <numeric>", "#include <cmath>",
    "#include <cstdint>", "#include <optional>", "#include <utility>", "#include <functional>", "#include <chrono>",
]

# Compiled after solution macros are #undef'd. `__ss::cmp` compares std/builtin values with its own element-wise
# equality (member calls + builtin `==` on scalars), so a solution's global `operator==` for std types is never
# consulted; the dummy operators below hide global-namespace overloads from unqualified lookup inside `__ss`.
# Custom (non-std) types still use their own operator==, but only against other custom types: a custom type
# compared with a plain value is rejected as dishonest.
_CPP_SUPPORT = r"""
static std::string __ss_esc(const std::string& s) { std::string o; for (unsigned char c : s) { if (c == '"' || c == '\\') { o += '\\'; o += static_cast<char>(c); } else if (c < 0x20) { o += ' '; } else { o += static_cast<char>(c); } } return o; }
namespace __ss {
struct never_t {};
bool operator==(never_t, never_t);
bool operator!=(never_t, never_t);
inline bool dishonest = false;
template <class T> struct cat_t : std::integral_constant<int, (std::is_arithmetic_v<T> || std::is_enum_v<T> || std::is_pointer_v<T> || std::is_null_pointer_v<T>) ? 1 : 0> {};
template <class C, class T, class A> struct cat_t<std::basic_string<C, T, A>> : std::integral_constant<int, 2> {};
template <class C, class T> struct cat_t<std::basic_string_view<C, T>> : std::integral_constant<int, 2> {};
template <class T, class A> struct cat_t<std::vector<T, A>> : std::integral_constant<int, 3> {};
template <class T, std::size_t N> struct cat_t<std::array<T, N>> : std::integral_constant<int, 3> {};
template <class T, class A> struct cat_t<std::deque<T, A>> : std::integral_constant<int, 3> {};
template <class T, class A> struct cat_t<std::list<T, A>> : std::integral_constant<int, 3> {};
template <class K, class C, class A> struct cat_t<std::set<K, C, A>> : std::integral_constant<int, 3> {};
template <class K, class C, class A> struct cat_t<std::multiset<K, C, A>> : std::integral_constant<int, 3> {};
template <class K, class V, class C, class A> struct cat_t<std::map<K, V, C, A>> : std::integral_constant<int, 3> {};
template <class K, class V, class C, class A> struct cat_t<std::multimap<K, V, C, A>> : std::integral_constant<int, 3> {};
template <class A, class B> struct cat_t<std::pair<A, B>> : std::integral_constant<int, 4> {};
template <class... T> struct cat_t<std::tuple<T...>> : std::integral_constant<int, 5> {};
template <class T> struct cat_t<std::optional<T>> : std::integral_constant<int, 6> {};
template <> struct cat_t<std::nullopt_t> : std::integral_constant<int, 7> {};
template <class K, class V, class H, class E, class A> struct cat_t<std::unordered_map<K, V, H, E, A>> : std::integral_constant<int, 8> {};
template <class K, class H, class E, class A> struct cat_t<std::unordered_set<K, H, E, A>> : std::integral_constant<int, 9> {};
template <class T> constexpr int cat = cat_t<std::decay_t<T>>::value;

template <class T> constexpr bool honest();
template <class... T> constexpr bool all_honest(std::tuple<T...>*) { return (honest<T>() && ...); }
template <class T> constexpr bool honest() {
    using U = std::decay_t<T>;
    constexpr int c = cat<U>;
    if constexpr (c == 1 || c == 2 || c == 7) return true;
    else if constexpr (c == 3 || c == 6 || c == 9) return honest<typename U::value_type>();
    else if constexpr (c == 8) return honest<typename U::key_type>() && honest<typename U::mapped_type>();
    else if constexpr (c == 4) return honest<typename U::first_type>() && honest<typename U::second_type>();
    else if constexpr (c == 5) return all_honest(static_cast<U*>(nullptr));
    else return false;
}

template <class T> auto scal(const T& x) { if constexpr (std::is_enum_v<T>) return static_cast<std::underlying_type_t<T>>(x); else return x; }
template <class A, class B> bool eq(const A& a, const B& b);
template <class A, class B, std::size_t... I> bool tup_eq(const A& a, const B& b, std::index_sequence<I...>) { return (eq(std::get<I>(a), std::get<I>(b)) && ...); }
template <class A, class B> bool eq(const A& a, const B& b) {
    constexpr int ca = cat<A>, cb = cat<B>;
    if constexpr (ca == 1 && cb == 1) return scal(a) == scal(b);
    else if constexpr ((ca == 2 || cb == 2) && std::is_constructible_v<std::string_view, const A&> && std::is_constructible_v<std::string_view, const B&>)
        return std::string_view(a).compare(std::string_view(b)) == 0;
    else if constexpr (ca == 3 && cb == 3) {
        auto ia = std::begin(a); auto ea = std::end(a); auto ib = std::begin(b); auto eb = std::end(b);
        for (; ia != ea && ib != eb; ++ia, ++ib) if (!eq(*ia, *ib)) return false;
        return ia == ea && ib == eb;
    }
    else if constexpr (ca == 4 && cb == 4) return eq(a.first, b.first) && eq(a.second, b.second);
    else if constexpr (ca == 5 && cb == 5) {
        if constexpr (std::tuple_size_v<A> == std::tuple_size_v<B>) return tup_eq(a, b, std::make_index_sequence<std::tuple_size_v<A>>{});
        else return false;
    }
    else if constexpr (ca == 6 && cb == 6) return a.has_value() == b.has_value() && (!a.has_value() || eq(*a, *b));
    else if constexpr (ca == 6 && cb == 7) return !a.has_value();
    else if constexpr (ca == 7 && cb == 6) return !b.has_value();
    else if constexpr (ca == 7 && cb == 7) return true;
    else if constexpr (ca == 6) return a.has_value() && eq(*a, b);
    else if constexpr (cb == 6) return b.has_value() && eq(a, *b);
    else if constexpr (ca == 8 && cb == 8) {
        if (a.size() != b.size()) return false;
        for (const auto& kv : a) { auto it = b.find(kv.first); if (it == b.end() || !eq(kv.second, it->second)) return false; }
        return true;
    }
    else if constexpr (ca == 9 && cb == 9) {
        if (a.size() != b.size()) return false;
        for (const auto& k : a) if (b.find(k) == b.end()) return false;
        return true;
    }
    else return static_cast<bool>(a == b);
}
template <bool EQ, class A, class B> bool cmp(const A& a, const B& b) {
    constexpr bool ha = honest<A>(), hb = honest<B>();
    if constexpr (ha && hb) { bool r = eq(a, b); return EQ ? r : !r; }
    else if constexpr (!ha && !hb) { if constexpr (EQ) return static_cast<bool>(a == b); else return static_cast<bool>(a != b); }
    else { dishonest = true; return false; }
}
}  // namespace __ss
"""
_CPP_KEYWORD_OPS = re.compile(r"\b(?:and|or|not|bitand|bitor|xor|compl|not_eq|and_eq|or_eq|xor_eq)\b")


def _split_cpp_eq(expr: str) -> tuple[str, str, str] | None:
    """Split `L == R` / `L != R` at its single top-level equality operator; None when unsure (falls back to the
    plain `static_cast<bool>(expr)` check). Bails on anything binding looser than `==` (&&, ||, ?:, =, &, |, ^, ,)."""
    if 'R"' in expr or _CPP_KEYWORD_OPS.search(expr):
        return None
    depth, i, n, hits = 0, 0, len(expr), []
    while i < n:
        c = expr[i]
        if c == "'" and 0 < i < n - 1 and expr[i - 1].isalnum() and expr[i + 1].isalnum():  # digit separator 1'000
            i += 1
            continue
        if c in "\"'":
            j = i + 1
            while j < n and expr[j] != c:
                j += 2 if expr[j] == "\\" else 1
            i = j + 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0:
            two = expr[i : i + 2]
            if two in ("==", "!=") and (i == 0 or expr[i - 1] not in "<>=!") and expr[i + 2 : i + 3] != "=":
                hits.append((i, two))
                i += 2
                continue
            if two in ("&&", "||", "<=", ">=", "<<", ">>", "->"):
                if two in ("&&", "||"):
                    return None
                i += 2
                continue
            if c in "?,&|^" or (c == "=" and expr[i + 1 : i + 2] != "="):
                return None
        i += 1
    if len(hits) != 1:
        return None
    pos, op = hits[0]
    lhs, rhs = expr[:pos].strip(), expr[pos + 2 :].strip()
    return (lhs, op, rhs) if lhs and rhs else None


def _cpp_assert_total(test_code: str) -> int:
    return max(1, sum(1 for line in test_code.splitlines() if _CPP_ASSERT_LINE.match(line)))


def _build_cpp_harness(test_code: str, undef_names: list[str] | tuple[str, ...] = ()) -> tuple[str, str, int]:
    """Return (headers, main_source, assert_count). `main_source` starts with #undef of solution macros."""
    includes, body = [], []
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
            split = _split_cpp_eq(ex)
            if split:
                lhs, op, rhs = split
                cond = f"[&]() -> bool {{ auto&& __ss_l = ({lhs}); auto&& __ss_r = ({rhs}); return __ss::cmp<{'true' if op == '==' else 'false'}>(__ss_l, __ss_r); }}()"
            else:
                cond = f"static_cast<bool>({ex})"
            body.append("    __ss_t += 1; __ss::dishonest = false;")
            body.append(
                f"    try {{ if ({cond}) {{ __ss_p += 1; }} else if (__ss_err == \"ok\") {{ __ss_err = std::string(__ss::dishonest ? \"DishonestComparison#{assert_count}: \" : \"AssertFailed#{assert_count}: \") + \"{escaped}\"; }} }} "
                f"catch (const std::exception& e) {{ if (__ss_err == \"ok\") __ss_err = std::string(\"Exception: \") + e.what(); }} "
                f"catch (...) {{ if (__ss_err == \"ok\") __ss_err = \"UnknownException#{assert_count}\"; }}"
            )
        else:
            body.append(f"    {line}")

    if assert_count == 0:
        assert_count = 1
        body.append("    __ss_t = 1; __ss_p = 1;")

    main_src = "\n".join([
        *(f"#undef {name}" for name in undef_names),
        _CPP_SUPPORT,
        "int main() {",
        "    std::string __ss_pfx; std::getline(std::cin, __ss_pfx);",
        "    int __ss_p = 0, __ss_t = 0;",
        "    std::string __ss_err = \"ok\";",
        "    auto __ss_t0 = std::chrono::steady_clock::now();",
        *body,
        "    double __ss_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - __ss_t0).count();",
        '    std::cout << __ss_pfx << "[" << __ss_p << "," << __ss_t << ",\\"" << __ss_esc(__ss_err) << "\\"," << __ss_ms << "]\\n";',
        "    return 0;",
        "}",
    ])
    return "\n".join([*_CPP_STD_HEADERS, *includes]), main_src, assert_count


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
    nonce = os.urandom(16).hex()
    prefix = f"{marker}{nonce}:"  # sent on stdin only; never written into any file the solution could read

    # ponytail: OS temp dir + AST/JS/CPP/SQL checks + stdin nonce secures single-tenant RLVR; add Docker/gVisor for multi-tenant SaaS.
    with tempfile.TemporaryDirectory(prefix="neutron_") as tmp:
        if is_cpp:
            cpp_units = []
            if files:
                patched, p_status = apply_search_replace_blocks(files, solution_text)
                if patched is None:
                    partial = 0.05 if p_status.startswith(("search_not_found", "ambiguous_search")) else 0.0
                    return partial, {"status": p_status, "passed": 0, "total": _cpp_assert_total(test_code), "ms": 0.0}
                err = _safe_write_tree(tmp, patched)
                if err:
                    return (-0.5 if err.startswith("banned_") else 0.0), {"status": err, "passed": 0, "total": _cpp_assert_total(test_code), "ms": 0.0}
                # #undef only macros the solution introduced; macros already in the task's files stay usable by tests.
                orig_macros = {m for src in files.values() for m in _CPP_DEFINE.findall(src)}
                new_macros = sorted({m for src in patched.values() for m in _CPP_DEFINE.findall(src)} - orig_macros)
                headers_src, main_src, expected_total = _build_cpp_harness(test_code, new_macros)
                cpp_units = [os.path.join(tmp, os.path.normpath(p)) for p in patched if p.endswith((".cpp", ".cc", ".cxx"))]
                entry = os.path.join(tmp, "__ss_main.cpp")
                with open(entry, "w", encoding="utf-8") as f:
                    f.write(f"{headers_src}\n\n{main_src}\n")
                cpp_units.append(entry)
            else:
                code = extract_code(solution_text)
                if not code:
                    return -0.5, {"status": "empty", "passed": 0, "total": _cpp_assert_total(test_code), "ms": 0.0}
                v = check_cpp_safety(code)
                if v:
                    return -0.5, {"status": v, "passed": 0, "total": _cpp_assert_total(test_code), "ms": 0.0}
                new_macros = sorted(set(_CPP_DEFINE.findall(code)) - set(_CPP_DEFINE.findall(starter_code or "")))
                headers_src, main_src, expected_total = _build_cpp_harness(test_code, new_macros)
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
            if is_js or is_ts:
                prologue, harness_src, expected_total = _build_node_harness(test_code, is_ts=is_ts)
            else:
                prologue = _PY_PROLOGUE
                harness_src, expected_total = _build_py_harness(test_code)
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
                    f.write(f"{prologue}\n{harness_src}\n")
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
                    v = check_py_shadowing(code)
                    if v:
                        return 0.0, {"status": v, "passed": 0, "total": expected_total, "ms": 0.0}
                full_code = f"{starter_code}\n\n{code}".strip() if starter_code else code
                futures = ""
                if not (is_js or is_ts):
                    futures, full_code = _hoist_future(full_code)
                entry = os.path.join(tmp, f"__ss_check{ext}")
                with open(entry, "w", encoding="utf-8") as f:
                    f.write(f"{futures}{prologue}\n{full_code}\n\n{harness_src}\n")

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
            proc = subprocess.run(run_cmd, cwd=tmp, env=env, input=prefix + "\n", capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        except subprocess.TimeoutExpired:
            return 0.0, {"status": "timeout", "passed": 0, "total": expected_total, "ms": round(timeout * 1000, 1)}

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(prefix):
            try:
                passed, total, err, algo_ms = json.loads(line[len(prefix):])
                ok = type(passed) is int and type(total) is int and 0 <= passed <= total == expected_total
                ok = ok and isinstance(algo_ms, (int, float)) and algo_ms == algo_ms and algo_ms >= 0  # not NaN
            except (ValueError, TypeError):
                ok = False
            if not ok:
                return 0.0, {"status": "harness_report_corrupt", "passed": 0, "total": expected_total, "ms": elapsed_ms}
            frac = passed / total if total else 0.0
            reward = round(0.1 + 0.9 * frac, 4) if passed < total else 1.0
            return reward, {"status": str(err), "passed": passed, "total": total, "ms": elapsed_ms, "algo_ms": algo_ms}

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
        score = 0.05 if (THINK_BLOCK.search(text) or "</think>" in text) else 0.0  # template may open <think>
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
