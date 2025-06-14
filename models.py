"""
Data models and classes for RAG Document Summarization System
"""

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ChunkInfo:
    """Information about a document chunk"""
    text: str
    start_idx: int
    end_idx: int
    chunk_id: int
    metadata: Dict[str, Any] = None

@dataclass
class RetrievalResult:
    """Result from vector retrieval"""
    chunk: ChunkInfo
    similarity_score: float
    rank: int

@dataclass
class SummaryResult:
    """Final summarization result"""
    summary: str
    retrieved_chunks: List[RetrievalResult]
    token_usage: Dict[str, int]
    latency: float
    metadata: Dict[str, Any]