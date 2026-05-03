"""
Lightweight observability helpers.

This module gives the workflow a span-like interface for logs and tests, while
keeping the public surface compatible with an OpenTelemetry/Phoenix adapter.
"""

import logging
from typing import Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class Tracer:
    """Span-like logger used by the underwriting workflow."""
    
    def __init__(self, name=None):
        self.name = name
    
    def start_as_current_span(self, operation_name):
        """Start a lightweight span."""
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def set_attribute(self, key, value):
        """Log an attribute on the lightweight span."""
        logger.info("trace attribute: %s = %s", key, value)
    
    def set_status(self, status):
        """Log a status on the lightweight span."""
        logger.info("trace status: %s", status)

def get_tracer(name=None):
    """Get the tracer for workflow tracking."""
    return Tracer(name)

def record_workflow_latency(workflow_name: str, duration_ms: float):
    """Log workflow latency locally."""
    logger.info("trace workflow %s completed in %.2fms", workflow_name, duration_ms)

class WorkflowTracer:
    """Context manager that records workflow timing."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = datetime.now()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds() * 1000
        record_workflow_latency(self.name, duration)
