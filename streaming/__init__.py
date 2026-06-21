"""Real-time run-event monitoring (Tier 4).

The in-domain version of the video's streaming track: instead of a voice or
vision pipeline, this streams the underwriting workflow's own run events, detects
anomalies in real time, decomposes per-request latency into a budget, explains
anomalies (with graceful degradation), and supports replay for debugging.
"""

from streaming.latency_budget import StageBudget, decompose_latency
from streaming.stream_monitor import Anomaly, AnomalyThresholds, MetricSnapshot, StreamMonitor

__all__ = [
    "StageBudget",
    "decompose_latency",
    "Anomaly",
    "AnomalyThresholds",
    "MetricSnapshot",
    "StreamMonitor",
]
