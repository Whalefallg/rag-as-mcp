"""Deterministic, dependency-free feature hashing embeddings for offline baselines."""
from __future__ import annotations
import hashlib, math, re
from typing import List
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import register_embedding

@register_embedding("local_hash")
class LocalHashEmbedding(BaseEmbedding):
    def __init__(self, model="feature-hash-384-v1", **kwargs):
        super().__init__(model=model, **kwargs); self.dimensions=384
    def embed(self,texts:List[str],trace=None)->List[List[float]]:
        self._validate_texts(texts); return [self._one(text) for text in texts]
    def _one(self,text):
        words=re.findall(r"[a-z0-9]+",text.lower()); features=words+[" ".join(words[i:i+2]) for i in range(len(words)-1)]
        vector=[0.0]*self.dimensions
        for feature in features:
            raw=hashlib.blake2b(feature.encode(),digest_size=8).digest(); value=int.from_bytes(raw,"big"); idx=value%self.dimensions; vector[idx] += -1.0 if value&(1<<63) else 1.0
        norm=math.sqrt(sum(v*v for v in vector)) or 1.0
        return [v/norm for v in vector]
