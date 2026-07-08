import logging
from typing import List, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, PointStruct,
    Prefetch, FusionQuery, Fusion, SparseVector,
)
from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
from src.utils import creer_modele_embedding
from src.config import NOM_VECTEUR_DENSE, NOM_VECTEUR_SPARSE, MODELE_SPARSE

logger = logging.getLogger(__name__)

_PAYLOAD_TEXT_KEYS = ("page_content", "text", "content", "page_text")


def _extraire_texte_payload(payload: dict) -> str:
    for key in _PAYLOAD_TEXT_KEYS:
        val = payload.get(key)
        if val and isinstance(val, str) and val.strip():
            return val
    meta = payload.get("metadata", {})
    if isinstance(meta, dict):
        for key in _PAYLOAD_TEXT_KEYS:
            val = meta.get(key)
            if val and isinstance(val, str) and val.strip():
                return val
    return ""


def _extraire_metadata_payload(payload: dict) -> dict:
    meta = payload.get("metadata", {})
    if isinstance(meta, dict) and meta:
        return meta
    return {"source": payload.get("source", "Qdrant"), "page": payload.get("page", 0)}


def creer_modele_sparse() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=MODELE_SPARSE)


class QdrantRetrieverWrapper:

    def __init__(self, url: str, api_key: str, collection_name: str, embedding_function=None):
        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        self.collection_name = collection_name
        self.embedding_function = embedding_function or creer_modele_embedding()
        self.modele_sparse = creer_modele_sparse()

        try:
            info = self.client.get_collection(collection_name)
            self.points_count = info.points_count or 0
            logger.info(f"collection '{collection_name}' chargee, {self.points_count} points")
            if self.points_count == 0:
                logger.warning(f"collection '{collection_name}' vide, lancez l'ingestion")
        except Exception as e:
            raise RuntimeError(f"impossible d'acceder a la collection '{collection_name}': {e}")

    def _vecteur_sparse(self, texte: str) -> SparseVector:
        embedding = list(self.modele_sparse.embed([texte]))[0]
        return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        try:
            vecteur_dense = self.embedding_function.embed_query(query)
            vecteur_sparse = self._vecteur_sparse(query)
        except Exception as e:
            logger.error(f"erreur generation embedding requete: {e}")
            return []

        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(query=vecteur_dense, using=NOM_VECTEUR_DENSE, limit=k * 4),
                    Prefetch(query=vecteur_sparse, using=NOM_VECTEUR_SPARSE, limit=k * 4),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=k,
                with_payload=True,
            )
            hits = result.points
        except Exception as e:
            logger.error(f"erreur recherche qdrant: {e}")
            return []

        resultats = []
        for point in hits:
            payload = point.payload or {}
            doc = Document(
                page_content=_extraire_texte_payload(payload),
                metadata=_extraire_metadata_payload(payload),
            )
            # le score rrf n'est pas une distance cosinus, on le convertit en pseudo distance
            distance = 1.0 / (1.0 + point.score)
            resultats.append((doc, distance))
        return resultats

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]