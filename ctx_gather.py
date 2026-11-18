"""Editor-style repo context for Quark: what a plugin can know at the moment of a completion, picked by NAME.

The model can only call helpers it can see. This gathers, from the code the editor has (the current file as typed so
far, without the missing code, plus the other files of the repo):

  1. definitions of the names used around the cursor ("go to definition": their signatures, wherever they live),
  2. the members of the class the cursor is in, and of classes used near the cursor (`self.x` / `this.x` / `obj.m()`),
  3. an outline of the current file outside the visible window (signatures only),
  4. what the current file's imports provide (their top-level signatures),
  5. what files that import this one, or sit in the same folder, provide.

On the benchmark's holes that need a name defined elsewhere in the repo, this shows 104 of the 207 names needed at
4,000 characters and 129 at 8,000; v0.4's import-skeleton context showed 58. About 75 can't be found by name at all
(generic members like `size` / `map` reached through an inferred type, or names the missing code introduces).

Everything is a one-line signature (`def f(a, b) -> int:`, `export function f(a: A): B {...}`, `self.items = []`,
`int count;`), grouped per file as `<|file_sep|>path` blocks, best first, until the character budget is spent. The
current file's outline comes last, right before the prompt, under its own path.

Used by make_ctx_bench.py (the benchmark) and, for v0.5, by the training-data builder. Same code for both, and later
for the editor plugin: it never sees the missing code.
"""
import re
from dataclasses import dataclass, field

from fim import FIM_SEP
from quark_data_v04 import import_graph

KW = set("""if else elif for while return const let var function class new this true false null None True False self
def import from export default async await in of try catch finally throw typeof void and or not is lambda pass raise with
as yield struct auto int bool char double float std static inline template typename namespace using public private
protected override virtual unsigned size_t nullptr break continue switch case do sizeof constexpr noexcept operator
delete undefined string number any type interface extends implements super print len range dict list str set tuple
isinstance get set constructor value key args kwargs cls""".split())
_IDENT = re.compile(r"[A-Za-z_$][\w$]*")


@dataclass
class Sym:
    name: str
    line: int  # 0-based line of its header
    text: str  # one-line signature, with the original indentation
    kind: str  # class / func / field / var
    parent: str = ""  # enclosing class
    end: int = -1  # last line (classes only)


@dataclass
class FileSyms:
    syms: list = field(default_factory=list)
    by_name: dict = field(default_factory=dict)


def _one_line(lines: list[str], i: int, py: bool) -> str:
    """The header starting at line i, joined over continuation lines while its parentheses are open (max 8 lines).
    Only parentheses outside string literals count (a regex like /\\[x]/ must not pull in the next lines)."""
    s, depth, j = "", 0, i
    while j < len(lines) and j < i + 8:
        part = lines[j].rstrip() if j == i else " " + lines[j].strip()
        s += part
        bare = _STR.sub("", part)
        depth += bare.count("(") - bare.count(")")
        if depth <= 0:
            break
        j += 1
    s = re.sub(r"\s+", " ", s.strip())
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    return indent + s[:220]


# --- Python ----------------------------------------------------------------------------------------------------------
_PY_DEF = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)")
_PY_CLASS = re.compile(r"^(\s*)class\s+(\w+)")
_PY_SELF = re.compile(r"^(\s*)self\.(\w+)\s*(?::[^=]+)?=(?!=)")
_PY_VAR = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*(?::\s*[^=]+)?=(?!=)")
_PY_FIELD = re.compile(r"^(\s+)([A-Za-z_]\w*)\s*:\s*[^=]+(?:=.*)?$")


def py_symbols(text: str) -> list[Sym]:
    lines = text.split("\n")
    out, stack, seen_fields = [], [], set()  # stack: [indent, kind ("class"/"def"), name, Sym]
    in_str = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if in_str:  # inside a triple-quoted string
            if in_str in s:
                in_str = None
            continue
        for q in ('"""', "'''"):
            if s.count(q) == 1:
                in_str = q
        if not s or s.startswith(("#", ")", "]", "}")):
            continue
        ind = len(ln) - len(ln.lstrip())
        while stack and ind <= stack[-1][0]:
            stack.pop()[3].end = i - 1
        top = stack[-1] if stack else None
        cls = next((e for e in reversed(stack) if e[1] == "class"), None)
        if m := _PY_CLASS.match(ln):
            sym = Sym(m[2], i, _one_line(lines, i, True), "class", top[2] if top and top[1] == "class" else "")
            if top is None or top[1] == "class":
                out.append(sym)
            stack.append([ind, "class", m[2], sym])
        elif m := _PY_DEF.match(ln):
            sym = Sym(m[2], i, _one_line(lines, i, True), "func", top[2] if top and top[1] == "class" else "")
            if top is None or top[1] == "class":  # module functions and methods, not nested helpers
                out.append(sym)
            stack.append([ind, "def", m[2], sym])
        elif cls and top[1] == "def" and (m := _PY_SELF.match(ln)):
            if (cls[2], m[2]) not in seen_fields:
                seen_fields.add((cls[2], m[2]))
                out.append(Sym(m[2], i, ln.rstrip()[:140], "field", cls[2]))
        elif top is None and (m := _PY_VAR.match(ln)):
            out.append(Sym(m[2], i, ln.rstrip()[:140], "var"))
        elif top and top[1] == "class" and ((m := _PY_FIELD.match(ln)) or (m := _PY_VAR.match(ln))):
            if (top[2], m[2]) not in seen_fields:
                seen_fields.add((top[2], m[2]))
                out.append(Sym(m[2], i, ln.rstrip()[:140], "field", top[2]))
    for e in stack:
        e[3].end = len(lines) - 1
    return out


# --- JS / TS / C++ (brace languages) ---------------------------------------------------------------------------------
_STR = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`")
_JS_TOP = re.compile(r"^(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?"
                     r"(function\*?|class|interface|type|enum|namespace|module|const|let|var)\s+([A-Za-z_$][\w$]*)")
_JS_MEMBER = re.compile(r"^(?:(?:public|private|protected|static|async|readonly|override|abstract|declare|get|set|"
                        r"accessor)\s+)*\*?(#?[A-Za-z_$][\w$]*)\s*[?!]?\s*(?:<[^>]*>\s*)?[(:=;]")
_CPP_TYPE = re.compile(r"^(?:template\s*<.*>\s*)?(?:typedef\s+)?(class|struct|union|enum(?:\s+class)?)\s+"
                       r"(?:\[\[[^\]]*\]\]\s*)?(?:[A-Z_]+\s+)?([A-Za-z_]\w*)")
_CPP_USING = re.compile(r"^(?:template\s*<.*>\s*)?using\s+([A-Za-z_]\w*)\s*=")
_CPP_DEFINE = re.compile(r"^#\s*define\s+([A-Za-z_]\w*)")
_CPP_FUNC = re.compile(r"(~?[A-Za-z_]\w*)\s*(?:<[^<>()]*>)?\s*\(")
_CPP_FIELD = re.compile(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)?(?:=[^=]|\{[^}]*\}\s*;|;)")
_NOT_NAME = KW | {"if", "for", "while", "switch", "return", "catch", "sizeof", "decltype", "static_assert", "alignof",
                  "noexcept", "requires", "throw", "defined", "assert", "co_await", "co_return", "co_yield"}


def brace_symbols(text: str, cpp: bool) -> list[Sym]:
    lines = text.split("\n")
    out, stack = [], []  # stack: one entry per open brace: [kind, name, sym]
    in_comment = False
    for i, ln in enumerate(lines):
        code = ln
        if in_comment:
            if "*/" not in code:
                continue
            code, in_comment = code[code.index("*/") + 2:], False
        code = _STR.sub('""', code)
        code = re.sub(r"/\*.*?\*/", "", code)
        if "/*" in code:
            code, in_comment = code[:code.index("/*")], True
        code = code.split("//")[0]
        s = code.strip()
        kinds = [k for k, _, _ in stack]
        top_level = all(k == "ns" for k in kinds)
        member_of = stack[-1][1] if kinds and kinds[-1] == "class" and all(k == "ns" for k in kinds[:-1]) else None
        decl_kind, sym = "block", None
        if s and not s.startswith(("}", "return ", "#include", "import ", "//")):
            if top_level:
                if not cpp and (m := _JS_TOP.match(s)):
                    kind = {"class": "class", "interface": "class", "enum": "class", "namespace": "ns",
                            "module": "ns"}.get(m[1], "func" if m[1].startswith("function") else "var")
                    if kind == "ns":
                        decl_kind = "ns"
                    else:
                        sym = Sym(m[2], i, _one_line(lines, i, False), kind)
                        decl_kind = "class" if kind == "class" else "block"
                elif cpp:
                    if re.match(r"^(?:inline\s+)?namespace\b|^extern\s+\"", s):
                        decl_kind = "ns"
                    elif (m := _CPP_TYPE.match(s)) and not s.endswith(";"):
                        sym, decl_kind = Sym(m[2], i, _one_line(lines, i, False), "class"), "class"
                    elif (m := _CPP_USING.match(s)) or (m := _CPP_DEFINE.match(s)):
                        sym = Sym(m[1], i, _one_line(lines, i, False), "var")
                    elif "(" in s and not s.startswith(("template", "#")) and "=" not in s.split("(")[0]:
                        m = _CPP_FUNC.search(s)
                        if m and m[1] not in _NOT_NAME:
                            parent = ""
                            q = re.search(r"([A-Za-z_]\w*)::~?" + re.escape(m[1].lstrip("~")) + r"\s*(?:<[^<>()]*>)?\s*\(", s)
                            if q:
                                parent = q[1]
                            sym = Sym(m[1].lstrip("~") if not q else m[1], i, _one_line(lines, i, False), "func",
                                      parent)
            elif member_of is not None:
                if cpp:
                    if re.match(r"^(?:public|private|protected)\s*:", s) or s.startswith(("template", "#", "friend")):
                        pass
                    elif (m := _CPP_TYPE.match(s)) and not s.endswith(";"):
                        sym, decl_kind = Sym(m[2], i, _one_line(lines, i, False), "class", member_of), "class"
                    elif (m := _CPP_USING.match(s)):
                        sym = Sym(m[1], i, _one_line(lines, i, False), "var", member_of)
                    elif "(" in s and "=" not in s.split("(")[0]:
                        m = _CPP_FUNC.search(s)
                        if m and m[1] not in _NOT_NAME:
                            sym = Sym(m[1], i, _one_line(lines, i, False), "func", member_of)
                    elif (m := _CPP_FIELD.search(s)) and m[1] not in _NOT_NAME:
                        sym = Sym(m[1], i, ln.rstrip()[:140], "field", member_of)
                elif (m := _JS_MEMBER.match(s)) and m[1] not in _NOT_NAME:
                    sym = Sym(m[1], i, _one_line(lines, i, False), "func" if "(" in s[:s.find(m[1]) + len(m[1]) + 4]
                              else "field", member_of)
        if sym:
            out.append(sym)
        for ch in code:
            if ch == "{":
                stack.append([decl_kind, sym.name if sym else "", sym])
                decl_kind, sym = "block", None  # only the first brace on the line belongs to the declaration
            elif ch == "}" and stack:
                k, _, owner = stack.pop()
                if owner is not None and k == "class":
                    owner.end = i
    for k, _, owner in stack:
        if owner is not None and k == "class":
            owner.end = len(lines) - 1
    return out


def symbols(path: str, text: str) -> list[Sym]:
    if path.endswith(".py"):
        return py_symbols(text)
    return brace_symbols(text, cpp=path.endswith((".h", ".hpp", ".hh", ".hxx", ".cpp", ".cc", ".cxx")))


def _index(syms: list[Sym]) -> FileSyms:
    fs = FileSyms(syms)
    for s in syms:
        fs.by_name.setdefault(s.name, []).append(s)
    return fs


def _render(sym: Sym, py: bool) -> str:
    t = sym.text.rstrip()
    if not py and t.endswith("{"):
        t = t[:-1].rstrip() + " {...}"
    return t


class RepoContext:
    """Symbols of every repo file, built once per repo; `gather` then picks context for one cursor position."""

    def __init__(self, files: dict[str, str]):
        self.files = files
        self.syms = {f: _index(symbols(f, t)) for f, t in files.items()}
        self.graph = import_graph(files)
        self.callers = {}
        for f, deps in self.graph.items():
            for d in deps:
                self.callers.setdefault(d, set()).add(f)
        self.defs = {}  # name -> [(file, Sym)]
        for f, fs in self.syms.items():
            for s in fs.syms:
                self.defs.setdefault(s.name, []).append((f, s))

    def gather(self, rel: str, editor_text: str, win_lo: int, win_hi: int, cursor: int, budget: int = 4000) -> str:
        """Context for a completion in `rel`. `editor_text` is the file as the editor has it (the missing code not
        there); the prompt shows its characters [win_lo, win_hi) and the cursor is at `cursor`."""
        py = rel.endswith(".py")
        lo_line = editor_text.count("\n", 0, win_lo)
        hi_line = editor_text.count("\n", 0, win_hi)
        cur_line = editor_text.count("\n", 0, cursor)
        here = _index(symbols(rel, editor_text))
        outside = [s for s in here.syms if not lo_line <= s.line <= hi_line]
        deps = import_graph({**{f: "" for f in self.files if f != rel}, rel: editor_text}).get(rel, set())

        window = editor_text[win_lo:win_hi]
        rel_cur = cursor - win_lo
        dist = {}
        for m in _IDENT.finditer(window):
            name = m[0]
            if len(name) < 3 or name in KW:
                continue
            d = abs(m.start() - rel_cur)
            dist[name] = min(d, dist.get(name, d))
        near = sorted(dist, key=dist.get)

        picked = {}  # (file, line) -> (priority, order, Sym)
        order = [0]

        def add(f, s, prio):
            key = (f, s.line)
            if key not in picked or picked[key][0] > prio:
                picked[key] = (prio, order[0], s)
                order[0] += 1

        def members(f, fs, cls_name, prio, skip_window=False):
            for s in fs.syms:
                if s.parent == cls_name and not (skip_window and lo_line <= s.line <= hi_line):
                    add(f, s, prio)

        # 2a. the class the cursor is in: its members outside the window (self.x / this.x)
        enclosing = [s for s in here.syms if s.kind == "class" and s.line <= cur_line <= s.end]
        for c in enclosing[-1:]:
            if not lo_line <= c.line <= hi_line:
                add(rel, c, 0)
            members(rel, here, c.name, 1, skip_window=True)
        # 1. definitions of names near the cursor; classes bring their members (2b)
        for rank, name in enumerate(near[:60]):
            cands = [(rel, s) for s in here.by_name.get(name, []) if s in outside]
            others = [(f, s) for f, s in self.defs.get(name, []) if f != rel]
            others.sort(key=lambda fs: (fs[0] not in deps, fs[0] not in self.callers.get(rel, set())))
            for f, s in (cands + others)[:2]:
                add(f, s, 2 + rank / 100)
                if s.kind == "class":
                    members(f, here if f == rel else self.syms[f], s.name, 3 + rank / 100,
                            skip_window=f == rel)
                elif s.parent:  # a method: show its class header too
                    fs = here if f == rel else self.syms[f]
                    for c in fs.by_name.get(s.parent, []):
                        if c.kind == "class":
                            add(f, c, 2 + rank / 100)
        # 4. what the imports provide (ranked after the outline below)
        for f in sorted(deps):
            for s in self.syms[f].syms:
                if not s.parent or s.parent in dist:
                    add(f, s, 4)
        # 3. the current file outside the window
        for s in outside:
            add(rel, s, 3.5 if not s.parent else 3.8)
        # 5. importers and same-folder files
        folder = rel.rsplit("/", 1)[0] if "/" in rel else ""
        for f in sorted(self.callers.get(rel, set())) + sorted(
                p for p in self.files if p != rel and (p.rsplit("/", 1)[0] if "/" in p else "") == folder):
            for s in self.syms[f].syms:
                if not s.parent:
                    add(f, s, 7)

        # Fill the budget best-first, then print each file's picks in file order.
        chosen, used, headers = {}, 0, set()
        for (f, line), (prio, _, s) in sorted(picked.items(), key=lambda kv: kv[1][:2]):
            cost = len(_render(s, py)) + 1 + (len(f) + 14 if f not in headers else 0)
            if used + cost > budget:
                continue
            chosen.setdefault(f, []).append(s)
            headers.add(f)
            used += cost
        if not chosen:
            return ""
        files = sorted((f for f in chosen if f != rel), key=lambda f: min(picked[(f, s.line)][0] for s in chosen[f]),
                       reverse=True)  # the most relevant file sits closest to the prompt
        blocks = []
        for f in files + ([rel] if rel in chosen else []):
            body = "\n".join(_render(s, f.endswith(".py")) for s in sorted(chosen[f], key=lambda s: s.line))
            blocks.append(f"{FIM_SEP}{f}\n{body}\n")
        return "".join(blocks) + f"{FIM_SEP}{rel}\n"
