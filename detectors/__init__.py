"""
Fake News Detection Modules

This package contains different detection strategies:
- MLDetector: Traditional ML approach (TF-IDF + LinearSVC)
- LLMDetector: LangChain + LLM approach
- HybridDetector: Combines both for optimal performance
"""

from .ml_detector import MLDetector
from .llm_detector import LLMDetector
from .hybrid_detector import HybridDetector

__all__ = ['MLDetector', 'LLMDetector', 'HybridDetector']

