class BaseReranker:
    """Base class for all rerankers."""
    def rerank(self, affiliation, candidates):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

