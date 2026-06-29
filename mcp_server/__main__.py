"""Stdio entrypoint: `python -m mcp_server`.

Runs the read-only MCP server over stdio, the transport MCP clients (Claude
Desktop, Claude Code) use for local servers.
"""

from mcp_server.server import mcp

if __name__ == "__main__":
    mcp.run()
