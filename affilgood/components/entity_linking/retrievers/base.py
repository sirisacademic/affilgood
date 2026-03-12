"""
Base classes for entity linking retrievers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    """
    A candidate match returned by a retriever.

    Attributes
    ----------
    id : str
        Registry identifier (e.g. "https://ror.org/052gg0110").
    name : str
        Canonical name from the registry.
    score : float
        Retrieval score (0–1, higher = better match).
    source : str
        Registry source ("ror", "wikidata", etc.).
    matched_text : str
        Which name variant produced the match
        (could be an alias, acronym, or label).
    retriever : str
        Which retriever found this ("dense", "sparse", "combined").
    metadata : dict
        Additional record metadata (country, city, types, etc.).
    """
    id: str
    name: str
    score: float
    source: str = ""
    matched_text: str = ""
    retriever: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "score": round(self.score, 4),
            "source": self.source,
            "matched_text": self.matched_text,
            "retriever": self.retriever,
            "metadata": self.metadata,
        }


class BaseRetriever(ABC):
    """
    Abstract base class for entity linking retrievers.

    All retrievers must implement `retrieve()`.

    Custom retrievers can be passed directly to EntityLinker:

        class MyRetriever(BaseRetriever):
            def retrieve(self, query, top_k=10, context=None):
                ...
                return [Candidate(...), ...]

        ag = AffilGood(
            enable_entity_linking=True,
            linking_config={"retriever": MyRetriever()},
        )
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Candidate]:
        """
        Retrieve candidate matches for a query string.

        Parameters
        ----------
        query : str
            Organization name to match.
        top_k : int
            Maximum number of candidates to return.
        context : dict, optional
            NER context for disambiguation:
            {"country": "Spain", "city": "Barcelona"}

        Returns
        -------
        list of Candidate
            Candidates sorted by score descending.
        """
        ...