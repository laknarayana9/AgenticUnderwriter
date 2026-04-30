"""
Local demo observability helpers.

This module intentionally does not provide production tracing. It gives the
workflow a span-like interface for local logs and tests, while keeping the
public surface compatible with a future OpenTelemetry/Phoenix adapter.
"""

import logging
from typing import Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class Tracer:
    """Local span-like logger used by the demo workflow."""
    
    def __init__(self, name=None):
        self.name = name
    
    def start_as_current_span(self, operation_name):
        """Start a local demo span."""
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def set_attribute(self, key, value):
        """Log an attribute on the local demo span."""
        logger.info("demo_trace attribute: %s = %s", key, value)
    
    def set_status(self, status):
        """Log a status on the local demo span."""
        logger.info("demo_trace status: %s", status)

def get_tracer(name=None):
    """Get the local demo tracer for workflow tracking."""
    return Tracer(name)

def record_workflow_latency(workflow_name: str, duration_ms: float):
    """Log workflow latency locally."""
    logger.info("demo_trace workflow %s completed in %.2fms", workflow_name, duration_ms)

class WorkflowTracer:
    """Context manager that records local demo workflow timing."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = datetime.now()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds() * 1000
        record_workflow_latency(self.name, duration)
