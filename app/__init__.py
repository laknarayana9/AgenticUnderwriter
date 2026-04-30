#!/usr/bin/env python3
"""
Agentic Underwriting Platform - Core Module

This package implements the core components of the underwriting system.
"""

__version__ = "1.0.0"
__title__ = "Agentic Underwriting Platform"
__description__ = "Underwriting system with agent-based processing"
__author__ = "Agentic Underwriter contributors"

# Core components
from .rag_engine import get_rag_engine, RAGEngine

__all__ = [
    "get_rag_engine",
    "RAGEngine",
    "__version__",
    "__title__",
    "__description__",
    "__author__"
]
