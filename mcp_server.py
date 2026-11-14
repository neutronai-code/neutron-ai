"""NeutronAI MCP Server (`mcp_server.py`) — Expose NeutronAI Execution & Repo Tools Inside Any Coding Agent!

Why this is a massive force-multiplier for a lean AI lab:
  - Instead of building a custom VS Code extension from scratch, **Model Context Protocol (MCP)** lets
    OpenCode, Claude Code, Cursor, Windsurf, and Cline call NeutronAI's execution engine *natively* over stdio JSON-RPC 2.0!
  - Exposes 5 high-leverage developer tools (zero external dependencies):
    1. `neutron_verify`    : Execute & verify Python / JS / TypeScript / C++23 / SQL snippets or multi-file `SEARCH/REPLACE` patches with dense per-assert + speed scoring.
    2. `neutron_repomap`   : Build compact cross-file AST API skeletons + 1-hop import graphs for a repository.
    3. `neutron_localize`  : Rank the top-K relevant files in a local repository given a stack trace or bug description.
    4. `neutron_mine_bugs` : Generate execution-verified mutant bug-fix tasks (Py/JS/TS/C++) from a source + test snippet.
    5. `neutron_audit`     : Check prompt/test overlap (Jaccard contamination) before adding tasks to training data.

Add to OpenCode / Cursor / Claude MCP config:
  {
    "mcpServers": {
      "neutron-ai": {
        "command": "uv",
        "args": ["--directory", "/path/to/neutron-ai", "run", "mcp_server.py"]
      }
    }
  }

Usage:
  uv run mcp_server.py --self-check
"""
import argparse, json, sys
from agent import collect_repo_files, localize_files
from audit import decontaminate
from mutate import mine_bug_tasks
from repomap import build_repo_map
from sandbox import cache_stats, verify

SERVER_INFO = {"name": "neutron-mcp", "version": "0.9.0"}

TOOLS = [
    {
        "name": "neutron_verify",
        "description": "Execute & verify code or multi-file SEARCH/REPLACE patches in an isolated sandbox (python, javascript, typescript, cpp, sql). Returns reward (-0.5..1.0) and assertion diagnostics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "solution_text": {"type": "string", "description": "Model completion (`<think>` + fenced code or `### FILE:` SEARCH/REPLACE blocks)"},
                "test_code": {"type": "string", "description": "Unit test assertions"},
                "starter_code": {"type": "string", "default": ""},
                "files": {"type": "object", "description": "Optional `{rel_path: content}` map for multi-file repo patch verification"},
                "lang": {"type": "string", "enum": ["python", "javascript", "typescript", "cpp", "sql"], "default": "python"},
                "perf_budget_ms": {"type": "number", "default": 0.0, "description": "Optional algorithmic execution speed budget in ms"},
            },
            "required": ["solution_text", "test_code"],
        },
    },
    {
        "name": "neutron_repomap",
        "description": "Build a compact cross-file API skeleton map (classes, typed signatures, 1-hop import graph) for a repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_dir": {"type": "string"},
                "focus_files": {"type": "array", "items": {"type": "string"}},
                "max_chars": {"type": "integer", "default": 4000},
            },
            "required": ["repo_dir"],
        },
    },
    {
        "name": "neutron_localize",
        "description": "Rank the most relevant source files in a local repository given a failing stack trace or bug goal.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_dir": {"type": "string"},
                "query_and_trace": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["repo_dir", "query_and_trace"],
        },
    },
    {
        "name": "neutron_mine_bugs",
        "description": "Synthesize execution-verified bug-fix RL tasks + ground-truth oracle patches from clean source + tests via mutation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rel_path": {"type": "string", "default": "module.py"},
                "clean_source": {"type": "string"},
                "test_code": {"type": "string"},
                "max_tasks": {"type": "integer", "default": 5},
            },
            "required": ["clean_source", "test_code"],
        },
    },
    {
        "name": "neutron_audit",
        "description": "Decontaminate & deduplicate candidate training rows against held-out evaluation benchmark rows.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "train_rows": {"type": "array", "items": {"type": "object"}},
                "eval_rows": {"type": "array", "items": {"type": "object"}},
                "threshold": {"type": "number", "default": 0.5},
            },
            "required": ["train_rows", "eval_rows"],
        },
    },
]


def dispatch_tool(name: str, args: dict) -> dict:
    if name == "neutron_verify":
        score, info = verify(
            solution_text=args["solution_text"],
            test_code=args["test_code"],
            starter_code=args.get("starter_code", ""),
            files=args.get("files"),
            lang=args.get("lang", "python"),
            perf_budget_ms=float(args.get("perf_budget_ms", 0.0)),
        )
        return {"reward": score, **info, "cache": cache_stats()}
    if name == "neutron_repomap":
        files = collect_repo_files(args["repo_dir"])
        rmap = build_repo_map(files, args.get("focus_files"), max_chars=int(args.get("max_chars", 4000)))
        return {"repo_map": rmap, "total_indexed": len(files)}
    if name == "neutron_localize":
        files = collect_repo_files(args["repo_dir"])
        ranked = localize_files(files, args["query_and_trace"], top_k=int(args.get("top_k", 5)))
        return {"ranked_files": ranked, "total_indexed": len(files)}
    if name == "neutron_mine_bugs":
        tasks = mine_bug_tasks(
            args.get("rel_path", "module.py"),
            args["clean_source"],
            args["test_code"],
            max_tasks=int(args.get("max_tasks", 5)),
        )
        return {"count": len(tasks), "tasks": tasks}
    if name == "neutron_audit":
        clean, stats = decontaminate(args["train_rows"], args["eval_rows"], float(args.get("threshold", 0.5)))
        return {"stats": stats, "clean_rows": clean}
    raise ValueError(f"Unknown tool: {name}")


def handle_rpc(msg: dict) -> dict | None:
    method = msg.get("method", "")
    req_id = msg.get("id")
    params = msg.get("params") or {}

    # Notifications (no id) require no response in JSON-RPC 2.0
    if req_id is None:
        return None

    try:
        if method == "initialize":
            result = {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            out = dispatch_tool(params.get("name", ""), params.get("arguments") or {})
            result = {"content": [{"type": "text", "text": json.dumps(out)}], "isError": False}
        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True},
        }


def serve_stdio():
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        resp = handle_rpc(json.loads(line))
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


def run_self_check():
    init = handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
    assert init and init["result"]["serverInfo"]["name"] == "neutron-mcp", init

    listed = handle_rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert len(listed["result"]["tools"]) == 5, listed

    call = handle_rpc({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "neutron_verify",
            "arguments": {
                "lang": "sql",
                "starter_code": "CREATE TABLE items(price INT); INSERT INTO items VALUES (10), (30);",
                "solution_text": "```sql\nSELECT SUM(price) FROM items;\n```",
                "test_code": "assert rows == [(40,)]",
            },
        },
    })
    payload = json.loads(call["result"]["content"][0]["text"])
    assert payload["reward"] == 1.0 and payload["passed"] == 1, payload
    print("mcp_server.py (neutron-mcp) self-check passed: JSON-RPC 2.0 MCP initialize + 5 tools/list + SQL verification call.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        run_self_check()
    else:
        serve_stdio()
