import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from models.schemas import RunRecord, WorkflowState, HumanReviewRecord, QuoteRecord

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class UnderwritingDB:
    """
    SQLite database for storing underwriting run records.
    """
    
    def __init__(self, db_path: str = "storage/underwriting.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"🗄️ Initializing database at {self.db_path}")
        self.init_db()
    
    def init_db(self):
        """
        Initialize the database schema.
        """
        logger.info(" Initializing database schema")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS run_records (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    workflow_state TEXT NOT NULL,
                    node_outputs TEXT,
                    error_message TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_id ON run_records(run_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON run_records(created_at)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS human_review_records (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    requires_human_review BOOLEAN NOT NULL DEFAULT 1,
                    final_decision TEXT,
                    reviewer TEXT,
                    review_timestamp TEXT,
                    approved_premium REAL,
                    reviewer_notes TEXT,
                    review_priority TEXT,
                    assigned_reviewer TEXT,
                    estimated_review_time TEXT,
                    submission_timestamp TEXT,
                    review_deadline TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_review_run_id ON human_review_records(run_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_review_status ON human_review_records(status)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quote_records (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    message TEXT NOT NULL,
                    processing_time_ms INTEGER NOT NULL,
                    submission TEXT NOT NULL,
                    decision TEXT,
                    premium TEXT,
                    rce_adjustment TEXT,
                    requires_human_review BOOLEAN NOT NULL DEFAULT 0,
                    human_review_details TEXT,
                    required_questions TEXT,
                    citations TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quote_run_id ON quote_records(run_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quote_status ON quote_records(status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quote_timestamp ON quote_records(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON run_records(status)
            """)
            
            # Phase A Enhancement: Idempotency keys table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    idempotency_key TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response_run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_idempotency_user ON idempotency_keys(user_id)
            """)
            
            # Phase A Enhancement: Tool call event store (append-only)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    tool_call_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    latency_ms REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES run_records(run_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_calls_run_id ON tool_calls(run_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_calls_step ON tool_calls(step_name)
            """)
            
            # Phase A Enhancement: Retrieval events (for eval/debug)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_events (
                    retrieval_event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    filters_json TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    results_json TEXT NOT NULL,
                    latency_ms REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES run_records(run_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_retrieval_run_id ON retrieval_events(run_id)
            """)
            
            # Phase A Enhancement: HITL tasks
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hitl_tasks (
                    task_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    questions_json TEXT NOT NULL,
                    answers_json TEXT,
                    assigned_to TEXT,
                    sla_due_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES run_records(run_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hitl_run_id ON hitl_tasks(run_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hitl_status ON hitl_tasks(status)
            """)
            
            logger.info(" Database schema initialized successfully")
    
    def save_run_record(self, record: RunRecord) -> str:
        """
        Save a run record to the database.
        """
        logger.info(f"💾 Saving run record: {record.run_id}")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO run_records 
                (run_id, created_at, updated_at, status, workflow_state, node_outputs, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.run_id,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.status,
                record.workflow_state.model_dump_json(),
                json.dumps(record.node_outputs, cls=DateTimeEncoder),
                record.error_message
            ))
        
        return record.run_id
    
    def save_human_review_record(self, record: HumanReviewRecord) -> str:
        """
        Save a human review record to database.
        """
        def safe_isoformat(dt):
            return dt.isoformat() if dt else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO human_review_records 
                (run_id, status, requires_human_review, final_decision, reviewer, 
                 review_timestamp, approved_premium, reviewer_notes, review_priority, 
                 assigned_reviewer, estimated_review_time, submission_timestamp, review_deadline)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.run_id,
                record.status,
                record.requires_human_review,
                record.final_decision,
                record.reviewer,
                safe_isoformat(record.review_timestamp),
                record.approved_premium,
                record.reviewer_notes,
                record.review_priority,
                record.assigned_reviewer,
                record.estimated_review_time,
                safe_isoformat(record.submission_timestamp),
                safe_isoformat(record.review_deadline)
            ))
        
        return record.run_id
    
    def get_human_review_record(self, run_id: str) -> Optional[HumanReviewRecord]:
        """
        Retrieve a human review record by ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM human_review_records WHERE run_id = ?
            """, (run_id,))
            
            row = cursor.fetchone()
            if row:
                return HumanReviewRecord(
                    run_id=row["run_id"],
                    status=row["status"],
                    requires_human_review=bool(row["requires_human_review"]),
                    final_decision=row["final_decision"],
                    reviewer=row["reviewer"],
                    review_timestamp=datetime.fromisoformat(row["review_timestamp"]) if row["review_timestamp"] else None,
                    approved_premium=row["approved_premium"],
                    reviewer_notes=row["reviewer_notes"],
                    review_priority=row["review_priority"],
                    assigned_reviewer=row["assigned_reviewer"],
                    estimated_review_time=row["estimated_review_time"],
                    submission_timestamp=datetime.fromisoformat(row["submission_timestamp"]) if row["submission_timestamp"] else None,
                    review_deadline=datetime.fromisoformat(row["review_deadline"]) if row["review_deadline"] else None
                )
            return None
    
    def get_run_record(self, run_id: str) -> Optional[RunRecord]:
        """
        Retrieve a run record by ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM run_records WHERE run_id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            # Parse the data
            workflow_state = WorkflowState.model_validate_json(row['workflow_state'])
            node_outputs = json.loads(row['node_outputs']) if row['node_outputs'] else {}
            
            return RunRecord(
                run_id=row['run_id'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                status=row['status'],
                workflow_state=workflow_state,
                node_outputs=node_outputs,
                error_message=row['error_message']
            )
    
    def list_runs(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List recent runs with optional status filter.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT run_id, created_at, updated_at, status FROM run_records"
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def update_run_status(self, run_id: str, status: str, error_message: Optional[str] = None):
        """
        Update the status of a run.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE run_records 
                SET status = ?, updated_at = ?, error_message = ?
                WHERE run_id = ?
            """, (
                status,
                datetime.now().isoformat(),
                error_message,
                run_id
            ))
    
    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run record.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM run_records WHERE run_id = ?",
                (run_id,)
            )
            return cursor.rowcount > 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about runs.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Total runs
            total_runs = conn.execute("SELECT COUNT(*) as count FROM run_records").fetchone()['count']
            
            # Runs by status
            status_counts = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM run_records 
                GROUP BY status
            """).fetchall()
            
            # Recent runs (last 24 hours)
            recent_runs = conn.execute("""
                SELECT COUNT(*) as count 
                FROM run_records 
                WHERE created_at > datetime('now', '-1 day')
            """).fetchone()['count']
            
            return {
                "total_runs": total_runs,
                "recent_runs_24h": recent_runs,
                "runs_by_status": {row['status']: row['count'] for row in status_counts}
            }
    
    def save_quote_record(self, record: QuoteRecord) -> str:
        """
        Save a quote record to database.
        """
        def safe_isoformat(dt):
            return dt.isoformat() if dt else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO quote_records 
                (run_id, status, timestamp, message, processing_time_ms, 
                 submission, decision, premium, rce_adjustment, requires_human_review,
                 human_review_details, required_questions, citations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.run_id,
                record.status,
                safe_isoformat(record.timestamp),
                record.message,
                record.processing_time_ms,
                json.dumps(record.submission),
                json.dumps(record.decision) if record.decision else None,
                json.dumps(record.premium) if record.premium else None,
                json.dumps(record.rce_adjustment) if record.rce_adjustment else None,
                record.requires_human_review,
                json.dumps(record.human_review_details) if record.human_review_details else None,
                json.dumps(record.required_questions) if record.required_questions else None,
                json.dumps(record.citations) if record.citations else None
            ))
        
        return record.run_id
    
    def get_quote_record(self, run_id: str) -> Optional[QuoteRecord]:
        """
        Retrieve a quote record by ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM quote_records WHERE run_id = ?", (run_id,)
            ).fetchone()
            
            if cursor:
                return QuoteRecord(
                    run_id=cursor["run_id"],
                    status=cursor["status"],
                    timestamp=datetime.fromisoformat(cursor["timestamp"]),
                    message=cursor["message"],
                    processing_time_ms=cursor["processing_time_ms"],
                    submission=json.loads(cursor["submission"]) if cursor["submission"] else {},
                    decision=json.loads(cursor["decision"]) if cursor["decision"] else None,
                    premium=json.loads(cursor["premium"]) if cursor["premium"] else None,
                    rce_adjustment=json.loads(cursor["rce_adjustment"]) if cursor["rce_adjustment"] else None,
                    requires_human_review=bool(cursor["requires_human_review"]),
                    human_review_details=json.loads(cursor["human_review_details"]) if cursor["human_review_details"] else None,
                    required_questions=json.loads(cursor["required_questions"]) if cursor["required_questions"] else None,
                    citations=json.loads(cursor["citations"]) if cursor["citations"] else None
                )
            return None


# Global database instance
db = UnderwritingDB()


def get_db() -> UnderwritingDB:
    """Get database instance (lazy initialization pattern)."""
    return db


# Phase A Enhancement: Additional database methods
def check_idempotency_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
    """
    Check if an idempotency key exists.
    """
    with sqlite3.connect(self.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM idempotency_keys WHERE idempotency_key = ?",
            (idempotency_key,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def store_idempotency_key(self, idempotency_key: str, user_id: str, request_hash: str, response_run_id: str):
    """
    Store an idempotency key.
    """
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO idempotency_keys
            (idempotency_key, user_id, request_hash, response_run_id, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            idempotency_key,
            user_id,
            request_hash,
            response_run_id,
            datetime.now().isoformat()
        ))


def save_tool_call(self, tool_call_id: str, run_id: str, step_name: str, tool_name: str,
                   input_json: str, output_json: Optional[str], status: str,
                   error_message: Optional[str] = None, latency_ms: Optional[float] = None):
    """
    Save a tool call event.
    """
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("""
            INSERT INTO tool_calls
            (tool_call_id, run_id, step_name, tool_name, input_json, output_json, status, error_message, latency_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tool_call_id,
            run_id,
            step_name,
            tool_name,
            input_json,
            output_json,
            status,
            error_message,
            latency_ms,
            datetime.now().isoformat()
        ))


def save_retrieval_event(self, retrieval_event_id: str, run_id: str, query_text: str,
                         filters_json: str, top_k: int, results_json: str, latency_ms: float):
    """
    Save a retrieval event.
    """
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("""
            INSERT INTO retrieval_events
            (retrieval_event_id, run_id, query_text, filters_json, top_k, results_json, latency_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            retrieval_event_id,
            run_id,
            query_text,
            filters_json,
            top_k,
            results_json,
            latency_ms,
            datetime.now().isoformat()
        ))


def create_hitl_task(self, task_id: str, run_id: str, status: str, priority: str,
                    questions_json: str, assigned_to: Optional[str] = None,
                    sla_due_at: Optional[str] = None):
    """
    Create a HITL task.
    """
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("""
            INSERT INTO hitl_tasks
            (task_id, run_id, status, priority, questions_json, assigned_to, sla_due_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_id,
            run_id,
            status,
            priority,
            questions_json,
            assigned_to,
            sla_due_at,
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))


def get_hitl_task(self, task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a HITL task by ID.
    """
    with sqlite3.connect(self.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM hitl_tasks WHERE task_id = ?",
            (task_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def list_hitl_tasks(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    List HITL tasks, optionally filtered by status.
    """
    with sqlite3.connect(self.db_path) as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM hitl_tasks"
        params = []
        
        if status:
            query += " WHERE status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def process_hitl_action(self, task_id: str, action: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Process an action on a HITL task.
    """
    with sqlite3.connect(self.db_path) as conn:
        # Update task with answers and status
        action_type = action.get("action")
        answers = action.get("answers", {})
        
        if action_type == "approve":
            new_status = "approved"
        elif action_type == "override":
            new_status = "overridden"
        elif action_type == "request_info":
            new_status = "answered"
        else:
            new_status = "processed"
        
        conn.execute("""
            UPDATE hitl_tasks
            SET status = ?, answers_json = ?, assigned_to = ?, updated_at = ?
            WHERE task_id = ?
        """, (
            new_status,
            json.dumps(answers),
            user_id,
            datetime.now().isoformat(),
            task_id
        ))
        
        return {
            "task_id": task_id,
            "status": new_status,
            "action": action_type,
            "processed_by": user_id,
            "processed_at": datetime.now().isoformat()
        }


# Add methods to the UnderwritingDB class
UnderwritingDB.check_idempotency_key = check_idempotency_key
UnderwritingDB.store_idempotency_key = store_idempotency_key
UnderwritingDB.save_tool_call = save_tool_call
UnderwritingDB.save_retrieval_event = save_retrieval_event
UnderwritingDB.create_hitl_task = create_hitl_task
UnderwritingDB.get_hitl_task = get_hitl_task
UnderwritingDB.list_hitl_tasks = list_hitl_tasks
UnderwritingDB.process_hitl_action = process_hitl_action
