"""
EMBEDDING SERVICE
Convert Text to Vector Embeddings.
"""

import hashlib
from openai import OpenAI
from typing import List, Dict
from backend.config import settings

class EmbeddingService:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.OPENROUTER_API_KEY)
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSIONS

        # In-memory cache to avoid re-embedding same text
        self.cache: Dict[str, List[float]] = {}

    def get_cash_key(self, text: str) -> str:
        """
        Generate Cache Key from Text
        Uses MDS hash to create consistent keys for identical text.
        """

        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def embed_text(self, text) -> List[float]:
        """
        Generate Embedding for a Single Text"
        """

        cache_key = self.get_cash_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format: "float"
            )

            embedding = response.data[0].embedding

            self.cache[cache_key] = embedding

            return embedding
        except Exception as e:
            print(f"[EMBEDDING ERROR]\t{str(e)}")
            raise
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates Embeddings for Multiple texts at once
        This is more efficient than calling embed_text() multiple times
        because it batches the API calls.
        """

        # check which texts are already cached
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):    
            cache_key = self.get_cash_key(text)
            if cache_key in self.cache:
                results[i] = self.cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if uncached_texts:
            try:
                BATCH_SIZE = 2048

                for batch_start in range(0, len(uncached_text), batch_size):
                    batch_end = min(batch_start + BATCH_SIZE, len(uncached_texts))
                    batch = uncached_texts[batch_start:batch_end]
                    
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                        encoding_format="float"
                    )

                    # store results and update cache
                    for i, data in enumerate(response.data):
                        global_idx = uncached_indices[batch_start + i]
                        embedding = data.embedding
                        results[global_idx] = embedding

                        # cache this embedding
                        cache_key = self.get_cash_key(batch[i])
                        self.cache[cache_key] = embedding
                    
            except Exception as e:
                print(f"[EMBEDDING BATCH ERROR]\t{str(e)}")
                raise
        
        return results
    
    def get_embedding_dimension(self) -> int:
        return self.dimension

    def clear_cache(self) -> None:
        self.cache.clear()
    
    def get_cache_size(self) -> int:
        return len(self.cache)

# global singleton instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """
    GET or CREATE the singleton instance of the EmbeddingService
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
