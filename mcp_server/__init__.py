"""Read-only MCP server for the governed underwriting system.

An observation surface: external agents can query runs, decisions, audit trails,
pending reviews, and reconstructed metrics — but there are deliberately NO tools
that submit quotes, approve reviews, or alter a decision. The agent-facing
surface inherits the same invariant as the rest of the system: the LLM/agent
never drives the governed decision.
"""
