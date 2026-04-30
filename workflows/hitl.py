"""
Minimal HITL (Human-in-the-Loop) workflow
"""

import logging
import json
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class HITLActionType(Enum):
    """HITL action types"""
    REVIEW = "review"
    APPROVE = "approve"
    REJECT = "reject"

def get_hitl_workflow(data=None):
    """Get HITL workflow instance"""
    return HITLWorkflow(db=data)

class HITLWorkflow:
    """Simple HITL workflow"""

    def __init__(self, db: Optional[Any] = None):
        self.db = db
    
    def create_hitl_task(self, run_id: str, task_type: str, description: str, **kwargs):
        """Create HITL task"""
        logger.info(f"Creating HITL task: {task_type} for run {run_id}")
        priority = kwargs.get("priority", "medium")
        questions = kwargs.get("metadata", {}).get("questions", [])
        question_key = "_".join(q.get("question_id", "review") for q in questions) or task_type
        task_id = kwargs.get("task_id") or f"task_{run_id}_{task_type}_{question_key}"
        if self.db and hasattr(self.db, "create_hitl_task"):
            self.db.create_hitl_task(
                task_id=task_id,
                run_id=run_id,
                status="open",
                priority=priority,
                questions_json=json.dumps(questions),
            )
        return {"task_id": task_id, "status": "created", "description": description}
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process HITL workflow"""
        logger.info("Processing HITL workflow")
        return {"hitl_status": "completed", "data": data}
