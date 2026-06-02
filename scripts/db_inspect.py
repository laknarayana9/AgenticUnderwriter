#!/usr/bin/env python3
"""
Inspect and verify data in the underwriting SQLite database.

This is a read-only helper for confirming what the API persisted. It resolves
the same database path the app uses (UNDERWRITING_DB_PATH / DATABASE_URL, else
storage/underwriting.db) and opens it in read-only mode.

Examples:
    # List every table with its row count
    python scripts/db_inspect.py tables

    # Show the CREATE statement / columns for a table
    python scripts/db_inspect.py schema run_records

    # Dump recent rows from a table (newest first when a timestamp exists)
    python scripts/db_inspect.py rows run_records --limit 5

    # Filter rows with a WHERE clause
    python scripts/db_inspect.py rows run_records --where "status = 'pending_review'"

    # Show everything tied to a single run_id across all tables
    python scripts/db_inspect.py run <run_id>

    # Run an arbitrary read-only SELECT
    python scripts/db_inspect.py query "SELECT status, COUNT(*) FROM run_records GROUP BY status"

    # Quick high-level stats
    python scripts/db_inspect.py stats

Add --json to any command to emit machine-readable output instead of a table.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "storage" / "underwriting.db"

# Tables that reference a run_id, used by the `run` command to gather everything
# associated with a single quote run.
RUN_SCOPED_TABLES = [
    "run_records",
    "human_review_records",
    "quote_records",
    "tool_calls",
    "retrieval_events",
    "hitl_tasks",
]


def resolve_db_path() -> Path:
    """Resolve the DB path the same way storage/database.py does."""
    explicit_path = os.getenv("UNDERWRITING_DB_PATH")
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    database_url = os.getenv("DATABASE_URL")
    if database_url and database_url.startswith("sqlite:///"):
        return Path(unquote(database_url.removeprefix("sqlite:///"))).expanduser().resolve()
    if database_url:
        parsed = urlparse(database_url)
        if parsed.scheme in ("", "sqlite"):
            path = unquote(parsed.path) or database_url
            return Path(path).expanduser().resolve()

    return DEFAULT_DB_PATH.resolve()


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the database read-only so inspection can never mutate state."""
    if not db_path.exists():
        sys.exit(f"Database not found at {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row["name"] for row in rows]


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row["name"] for row in rows]


def _truncate(value: object, width: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ")
    if len(text) > width:
        return text[: width - 1] + "…"
    return text


def _print_rows(rows: list[sqlite3.Row], as_json: bool, max_col_width: int = 48) -> None:
    if as_json:
        print(json.dumps([dict(row) for row in rows], indent=2, default=str))
        return

    if not rows:
        print("(no rows)")
        return

    columns = list(rows[0].keys())
    widths = {col: len(col) for col in columns}
    rendered: list[dict[str, str]] = []
    for row in rows:
        cells = {col: _truncate(row[col], max_col_width) for col in columns}
        rendered.append(cells)
        for col in columns:
            widths[col] = min(max(widths[col], len(cells[col])), max_col_width)

    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for cells in rendered:
        print("  ".join(cells[col].ljust(widths[col]) for col in columns))
    print(f"\n({len(rows)} row{'s' if len(rows) != 1 else ''})")


def cmd_tables(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    tables = _list_tables(conn)
    summary = []
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        summary.append({"table": table, "rows": count})

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    width = max((len(item["table"]) for item in summary), default=5)
    print(f"{'table'.ljust(width)}  rows")
    print(f"{'-' * width}  ----")
    for item in summary:
        print(f"{item['table'].ljust(width)}  {item['rows']}")


def cmd_schema(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    if args.table not in _list_tables(conn):
        sys.exit(f"Unknown table: {args.table}")
    info = conn.execute(f"PRAGMA table_info({args.table})").fetchall()
    if args.json:
        print(json.dumps([dict(row) for row in info], indent=2))
        return
    print(f"Schema for {args.table}:")
    for row in info:
        pk = " PRIMARY KEY" if row["pk"] else ""
        notnull = " NOT NULL" if row["notnull"] else ""
        print(f"  {row['name']} {row['type']}{notnull}{pk}")


def _order_clause(conn: sqlite3.Connection, table: str) -> str:
    columns = _table_columns(conn, table)
    for candidate in ("created_at", "timestamp", "updated_at", "submission_timestamp"):
        if candidate in columns:
            return f" ORDER BY {candidate} DESC"
    return ""


def cmd_rows(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    if args.table not in _list_tables(conn):
        sys.exit(f"Unknown table: {args.table}")

    query = f"SELECT * FROM {args.table}"
    if args.where:
        query += f" WHERE {args.where}"
    query += _order_clause(conn, args.table)
    query += " LIMIT ?"
    try:
        rows = conn.execute(query, (args.limit,)).fetchall()
    except sqlite3.Error as exc:
        sys.exit(f"Query failed: {exc}")
    _print_rows(rows, args.json)


def cmd_run(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    tables = _list_tables(conn)
    scoped: dict[str, list[sqlite3.Row]] = {}
    for table in RUN_SCOPED_TABLES:
        if table not in tables or "run_id" not in _table_columns(conn, table):
            continue
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE run_id = ?", (args.run_id,)
        ).fetchall()
        if rows:
            scoped[table] = rows

    if args.json:
        payload = {table: [dict(row) for row in rows] for table, rows in scoped.items()}
        print(json.dumps(payload, indent=2, default=str))
        return

    if not scoped:
        print(f"No records found for run_id={args.run_id}")
        return

    for table, rows in scoped.items():
        print(f"=== {table} ({len(rows)}) ===")
        _print_rows(rows, as_json=False)
        print()


def cmd_query(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    sql = args.sql.strip()
    lowered = sql.lower().lstrip("(")
    if not (lowered.startswith("select") or lowered.startswith("with") or lowered.startswith("pragma")):
        sys.exit("Only read-only SELECT/WITH/PRAGMA statements are allowed.")
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.Error as exc:
        sys.exit(f"Query failed: {exc}")
    _print_rows(rows, args.json)


def cmd_stats(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    stats: dict[str, object] = {}
    total = conn.execute("SELECT COUNT(*) AS c FROM run_records").fetchone()["c"]
    stats["total_runs"] = total
    by_status = conn.execute(
        "SELECT status, COUNT(*) AS c FROM run_records GROUP BY status ORDER BY c DESC"
    ).fetchall()
    stats["runs_by_status"] = {row["status"]: row["c"] for row in by_status}
    stats["pending_reviews"] = conn.execute(
        "SELECT COUNT(*) AS c FROM human_review_records WHERE status = 'pending_review'"
    ).fetchone()["c"]
    stats["open_hitl_tasks"] = conn.execute(
        "SELECT COUNT(*) AS c FROM hitl_tasks WHERE status = 'open'"
    ).fetchone()["c"]

    if args.json:
        print(json.dumps(stats, indent=2))
        return
    print(f"total_runs: {stats['total_runs']}")
    print("runs_by_status:")
    for status, count in stats["runs_by_status"].items():
        print(f"  {status}: {count}")
    print(f"pending_reviews: {stats['pending_reviews']}")
    print(f"open_hitl_tasks: {stats['open_hitl_tasks']}")


def build_parser() -> argparse.ArgumentParser:
    # Shared options attached to every subcommand. They are placed after the
    # subcommand, e.g. `db_inspect.py rows run_records --json`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="Override the database path (defaults to the app's configured path)")
    common.add_argument("--json", action="store_true", help="Emit JSON instead of a formatted table")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tables", help="List all tables with row counts", parents=[common])
    sub.add_parser("stats", help="Show high-level run/review stats", parents=[common])

    p_schema = sub.add_parser("schema", help="Show columns for a table", parents=[common])
    p_schema.add_argument("table")

    p_rows = sub.add_parser("rows", help="Dump rows from a table", parents=[common])
    p_rows.add_argument("table")
    p_rows.add_argument("--limit", type=int, default=20)
    p_rows.add_argument("--where", help="Optional WHERE clause (without the WHERE keyword)")

    p_run = sub.add_parser("run", help="Show all records for a single run_id", parents=[common])
    p_run.add_argument("run_id")

    p_query = sub.add_parser("query", help="Run a read-only SELECT", parents=[common])
    p_query.add_argument("sql")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve() if args.db else resolve_db_path()
    conn = connect(db_path)
    try:
        handlers = {
            "tables": cmd_tables,
            "schema": cmd_schema,
            "rows": cmd_rows,
            "run": cmd_run,
            "query": cmd_query,
            "stats": cmd_stats,
        }
        handlers[args.command](conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
