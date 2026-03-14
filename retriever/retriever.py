import logging
from storage.faiss_store import FAISSStore
from config import TOP_K_DEFAULT

logger = logging.getLogger(__name__)


class DDRRetriever:

    def __init__(self):
        self.store = FAISSStore()

    def load(self) -> bool:
        return self.store.load()

    def search(
        self,
        query:         str,
        top_k:         int = TOP_K_DEFAULT,
        filter_source: str = None,
        filter_type:   str = None,
    ) -> list[dict]:
        return self.store.search(query, top_k=top_k, filter_source=filter_source, filter_type=filter_type)

    def search_inspection(self, query: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        return self.search(query, top_k=top_k, filter_source="inspection")

    def search_thermal(self, query: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        return self.search(query, top_k=top_k, filter_source="thermal")

    def search_images(self, query: str, top_k: int = 3) -> list[dict]:
        return self.search(query, top_k=top_k, filter_type="image")

    def search_tables(self, query: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        return self.search(query, top_k=top_k, filter_type="table")

    def get_stats(self) -> dict:
        return self.store.get_stats()
