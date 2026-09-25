#!/usr/bin/env python3
"""Smoke-test a built rag-as-mcp container over MCP stdio."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def _request(method: str, request_id: int) -> dict:
    params = (
        {"protocolVersion": "2024-11-05"}
        if method == "initialize"
        else {}
    )
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Docker image tag to smoke-test")
    args = parser.parse_args()

    payload = "\n".join(
        json.dumps(req)
        for req in (
            _request("initialize", 1),
            _request("tools/list", 2),
        )
    ) + "\n"

    try:
        proc = subprocess.run(
            ["docker", "run", "--rm", "-i", args.image],
            input=payload,
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
        )
    except FileNotFoundError:
        print("ERROR: docker executable not found", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired:
        print("ERROR: container smoke test timed out", file=sys.stderr)
        return 3

    if proc.returncode != 0:
        print(
            f"ERROR: container exited with code {proc.returncode}",
            file=sys.stderr,
        )
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        return proc.returncode or 1

    responses = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            responses.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(
                f"ERROR: non-JSON stdout from MCP server: {line!r} ({exc})",
                file=sys.stderr,
            )
            return 4

    by_id = {response.get("id"): response for response in responses}

    init = by_id.get(1)
    if not init or "result" not in init:
        print(f"ERROR: initialize response missing/invalid: {init!r}", file=sys.stderr)
        return 5

    server_info = init["result"].get("serverInfo", {})
    if server_info.get("name") != "rag-as-mcp":
        print(f"ERROR: unexpected serverInfo: {server_info!r}", file=sys.stderr)
        return 6

    tools_response = by_id.get(2)
    if not tools_response or "result" not in tools_response:
        print(
            f"ERROR: tools/list response missing/invalid: {tools_response!r}",
            file=sys.stderr,
        )
        return 7

    tools = tools_response["result"].get("tools", [])
    tool_names = {tool.get("name") for tool in tools}
    required = {
        "query_knowledge_hub",
        "list_collections",
        "get_document_summary",
    }
    missing = sorted(required - tool_names)
    if missing:
        print(f"ERROR: missing MCP tools: {missing}", file=sys.stderr)
        return 8

    print(
        "Container MCP smoke test passed: "
        f"server={server_info.get('name')} tools={len(tools)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
