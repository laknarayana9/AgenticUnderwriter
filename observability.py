"""
Minimal observability module for workflow tracking
"""

import logging
from typing import Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class Tracer:
    """Simple tracer implementation"""
    
    def __init__(self, name=None):
        self.name = name
    
    def start_as_current_span(self, operation_name):
        """Start a span"""
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def set_attribute(self, key, value):
        """Set attribute on span"""
        logger.info(f"Setting attribute: {key} = {value}")
    
    def set_status(self, status):
        """Set status on span"""
        logger.info(f"Setting status: {status}")

def get_tracer(name=None):
    """Get tracer for workflow tracking"""
    return Tracer(name)

def record_workflow_latency(workflow_name: str, duration_ms: float):
    """Record workflow latency"""
    logger.info(f"Workflow {workflow_name} completed in {duration_ms}ms")

class WorkflowTracer:
    """Simple workflow tracer"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = datetime.now()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds() * 1000
        record_workflow_latency(self.name, duration)
