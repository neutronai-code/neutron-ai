"""NeutronAI Repository Intelligence (`repomap.py`) — Import-Graph & Symbol Skeleton Extractor.

Why Cursor, Aider, Windsurf, and Qwen2.5-Coder beat naive RAG on real codebases:
  1. **Import-Graph Localization**: When `app/paginator.py` crashes in a stack trace, the actual root-cause bug
     often lives in an imported helper (`app/math_util.py`) that NEVER appears in the traceback!
     Lexical BM25 misses 1-hop dependencies; `repomap.py` parses imports across Python, JS/TS, and C++
     and propagates localization signal across callers & callees.
  2. **AST / Signature Repo-Maps**: Dumping 50 raw files blows past context windows and wastes GPU VRAM.
     `build_repo_map()` compresses neighboring files into compact API skeletons (classes, function signatures,
     type annotations, exports) at ~10x fewer tokens—ideal for `<|file_sep|>` Repo-FIM & SWE-agent prompts!

Usage:
  uv run repomap.py --self-check
  uv run repomap.py --repo . --focus sandbox.py
"""
import argparse, ast, os, re

_JS_TS_IMPORT = re.compile(
    r"""(?:from\s+['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\s*\)|import\s*\(\s*['"]([^'"]+)['"]\s*\))"""
)
_CPP_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
_SIG_LINE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|const|struct|enum)\s+[A-Za-z_]"
)


def extract_file_skeleton(rel_path: str, content: str) -> str:
    """Return compact public API skeleton (`class` + `def` signatures with types; no bulky function bodies)."""
    if rel_path.endswith(".py"):
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return ""
        out = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_copy = ast.FunctionDef(
                    name=node.name, args=node.args, body=[ast.Expr(ast.Constant(...))],
                    decorator_list=[], returns=node.returns, lineno=1, col_offset=0,
                ) if isinstance(node, ast.FunctionDef) else ast.AsyncFunctionDef(
                    name=node.name, args=node.args, body=[ast.Expr(ast.Constant(...))],
                    decorator_list=[], returns=node.returns, lineno=1, col_offset=0,
                )
                ast.fix_missing_locations(fn_copy)
                out.append(ast.unparse(fn_copy))
            elif isinstance(node, ast.ClassDef):
                bases = f"({', '.join(ast.unparse(b) for b in node.bases)})" if node.bases else ""
                out.append(f"class {node.name}{bases}:")
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                for m in methods:
                    mc = ast.FunctionDef(
                        name=m.name, args=m.args, body=[ast.Expr(ast.Constant(...))],
                        decorator_list=[], returns=m.returns, lineno=1, col_offset=0,
                    )
                    ast.fix_missing_locations(mc)
                    for line in ast.unparse(mc).splitlines():
                        out.append(f"    {line}")
                if not methods:
                    out.append("    ...")
        return "\n".join(out)

    # JS / TS / C++ header & declaration skeleton
    sigs = [ln.rstrip() for ln in content.splitlines() if _SIG_LINE.match(ln)]
    return "\n".join(sigs[:25])


def _resolve_candidate(raw_ref: str, importer_rel: str, all_files: set[str]) -> str | None:
    base_dir = os.path.dirname(importer_rel).replace("\\", "/")
    candidates = []
    if raw_ref.startswith("."):
        joined = os.path.normpath(os.path.join(base_dir, raw_ref)).replace("\\", "/")
        candidates.extend([joined, *(f"{joined}{ext}" for ext in ("", ".py", ".js", ".ts", ".mts", ".cjs", ".hpp", ".h")), f"{joined}/index.ts", f"{joined}/index.js"])
    else:
        dotted = raw_ref.replace(".", "/")
        candidates.extend([
            raw_ref, dotted, f"{dotted}.py", f"{dotted}/__init__.py",
            os.path.normpath(os.path.join(base_dir, raw_ref)).replace("\\", "/"),
            *(f"{dotted}{ext}" for ext in (".js", ".ts", ".mts", ".hpp", ".h")),
        ])
    for c in candidates:
        if c in all_files and c != importer_rel:
            return c
    return None


def build_import_graph(files: dict[str, str]) -> dict[str, set[str]]:
    """Return `{file: {imported_repo_files}}` across Python, JS/TS, and C++."""
    known = set(files)
    graph: dict[str, set[str]] = {p: set() for p in files}
    for path, src in files.items():
        refs = []
        if path.endswith(".py"):
            try:
                tree = ast.parse(src)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        refs.extend(a.name for a in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        prefix = "." * (node.level or 0)
                        mod = node.module or ""
                        refs.append(f"{prefix}{mod}" if not prefix else os.path.join(os.path.dirname(path), mod.replace(".", "/")).replace("\\", "/"))
                        for a in node.names:
                            refs.append(f"{mod}.{a.name}" if mod else a.name)
            except SyntaxError:
                pass
        elif path.endswith((".js", ".ts", ".mts", ".cjs", ".mjs")):
            for m in _JS_TS_IMPORT.finditer(src):
                refs.append(next(g for g in m.groups() if g))
        elif path.endswith((".cpp", ".cc", ".cxx", ".hpp", ".h")):
            refs.extend(_CPP_INCLUDE.findall(src))

        for ref in refs:
            resolved = _resolve_candidate(ref, path, known)
            if resolved:
                graph[path].add(resolved)
    return graph


def graph_neighbors(files: dict[str, str], target_file: str) -> list[str]:
    """Return 1-hop imported callees first, then 1-hop callers of `target_file`."""
    g = build_import_graph(files)
    callees = sorted(g.get(target_file, ()))
    callers = sorted(p for p, deps in g.items() if target_file in deps and p != target_file)
    seen = {target_file}
    ordered = []
    for p in [*callees, *callers]:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def build_repo_map(files: dict[str, str], focus_files: list[str] | None = None, max_chars: int = 4000) -> str:
    """Format compact cross-file API skeleton map prioritizing `focus_files` and their 1-hop graph neighbors."""
    ordered = []
    seen = set()
    seeds = list(focus_files or [])
    for f in seeds:
        for nb in [f, *graph_neighbors(files, f)]:
            if nb in files and nb not in seen:
                seen.add(nb)
                ordered.append(nb)
    for p in sorted(files):
        if p not in seen:
            ordered.append(p)

    chunks, used = [], 0
    for p in ordered:
        skel = extract_file_skeleton(p, files[p])
        if not skel:
            continue
        block = f"# {p}\n{skel}"
        if used + len(block) > max_chars:
            break
        chunks.append(block)
        used += len(block)
    return "\n\n".join(chunks)


def run_self_check():
    repo_files = {
        "app/__init__.py": "",
        "app/math_util.py": "def ceil_div(total: int, size: int) -> int:\n    return (total + size - 1) // size\n",
        "app/paginator.py": "from app.math_util import ceil_div\n\ndef paginate(items: list, page: int, size: int) -> dict:\n    return {'total_pages': ceil_div(len(items), size)}\n",
        "app/unrelated.py": "class Noise:\n    def ping(self) -> str:\n        return 'pong'\n",
    }
    g = build_import_graph(repo_files)
    assert "app/math_util.py" in g["app/paginator.py"], g
    nbs = graph_neighbors(repo_files, "app/paginator.py")
    assert nbs == ["app/math_util.py"], nbs

    rmap = build_repo_map(repo_files, focus_files=["app/paginator.py"])
    assert "def ceil_div(total: int, size: int) -> int:" in rmap, rmap
    assert "(total + size - 1)" not in rmap, "Skeleton must strip implementation bodies!"
    print("repomap.py self-check passed: cross-file import graph + body-stripped API skeleton map verified.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--focus", default="")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        run_self_check()
    else:
        from agent import collect_repo_files
        fls = collect_repo_files(args.repo)
        print(build_repo_map(fls, [args.focus] if args.focus else None))
