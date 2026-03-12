"""
Base class for entity linking rerankers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..retrievers.base import Candidate


class BaseReranker(ABC):
    """
    Abstract base class for rerankers.

    Rerankers take a query + list of candidates and return
    a re-scored, re-sorted list.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Candidate]:
        """
        Rerank candidates for a query.

        Parameters
        ----------
        query : str
            The raw affiliation string or entity name.
        candidates : list of Candidate
            Retrieved candidates to rerank.
        context : dict, optional
            NER context: {"country": "Spain", "city": "Barcelona"}

        Returns
        -------
        list of Candidate
            Re-scored and sorted descending by score.
        """
        ...

    def free(self):
        """Release GPU memory (optional override)."""
        pass
