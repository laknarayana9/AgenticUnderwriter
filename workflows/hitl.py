"""
Minimal HITL (Human-in-the-Loop) workflow
"""

import logging
from typing import Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class HITLActionType(Enum):
    """HITL action types"""
    REVIEW = "review"
    APPROVE = "approve"
    REJECT = "reject"

def get_hitl_workflow(data=None):
    """Get HITL workflow instance"""
    return HITLWorkflow()

class HITLWorkflow:
    """Simple HITL workflow"""
    
    def create_hitl_task(self, run_id: str, task_type: str, description: str, **kwargs):
        """Create HITL task"""
        logger.info(f"Creating HITL task: {task_type} for run {run_id}")
        return {"task_id": f"task_{run_id}", "status": "created"}
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process HITL workflow"""
        logger.info("Processing HITL workflow")
        return {"hitl_status": "completed", "data": data}
