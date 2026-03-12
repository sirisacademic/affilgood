"""
Entity linking module for AffilGood.

Three-stage cascade pipeline:
  1. Direct match  (name+country → unique registry ID, ~35% coverage, ~98% precision)
  2. Dense retrieval + cross-encoder reranking with score fusion
  3. LLM listwise judge for low-confidence cases (optional)
"""

from .linker import EntityLinker, DirectMatcher
from .registry import RegistryRecord, RegistryManager
from .retrievers.base import BaseRetriever, Candidate
from .rerankers.base import BaseReranker
from .rerankers.cross_encoder import CrossEncoderReranker
from .rerankers.llm import LLMListwiseReranker

__all__ = [
    "EntityLinker",
    "DirectMatcher",
    "RegistryRecord",
    "RegistryManager",
    "BaseRetriever",
    "BaseReranker",
    "Candidate",
    "CrossEncoderReranker",
    "LLMListwiseReranker",
]
