import logging
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from src.utils import creer_modele_embedding

logger = logging.getLogger(__name__)

# le contenu textuel dans Qdrant
_PAYLOAD_TEXT_KEYS = ("page_content", "text", "content", "page_text")


def _extraire_texte_payload(payload: dict) -> str:
    
    for key in _PAYLOAD_TEXT_KEYS:
        val = payload.get(key)
        if val and isinstance(val, str) and val.strip():
            return val
    # chercher dans metadata imbriquée
    meta = payload.get("metadata", {})
    if isinstance(meta, dict):
        for key in _PAYLOAD_TEXT_KEYS:
            val = meta.get(key)
            if val and isinstance(val, str) and val.strip():
                return val
    logger.warning(
        f"Aucun texte trouvé dans le payload Qdrant. Clés disponibles : {list(payload.keys())}"
    )
    return ""


def _extraire_metadata_payload(payload: dict) -> dict:
    """
    Extrait les métadonnées du payload.
    Gère le cas où metadata est un dict imbriqué ou à plat.
    """
    meta = payload.get("metadata", {})
    if isinstance(meta, dict) and meta:
        return meta

    return {
        "source": payload.get("source", "Qdrant"),
        "page": payload.get("page", 0),
    }


class QdrantRetrieverWrapper:
    """Wrapper Qdrant."""

    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str,
        embedding_function=None,
    ):
        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        self.collection_name = collection_name
        self.embedding_function = embedding_function or creer_modele_embedding()
        self.search_kwargs = {}
        self.search_type = "similarity_score_threshold"

        try:
            info = self.client.get_collection(collection_name)
            self.points_count = info.points_count or 0

            vectors_cfg = info.config.params.vectors

            if isinstance(vectors_cfg, dict):
                first = next(iter(vectors_cfg.values()))
                self.vector_size = getattr(first, "size", None)
            elif hasattr(vectors_cfg, "size"):
                self.vector_size = vectors_cfg.size
            else:
                self.vector_size = None

            #  vérification réelle embedding
            try:
                test_vec = self.embedding_function.embed_query("test")
                embedding_size = len(test_vec)
            except Exception:
                embedding_size = None

            if self.vector_size and embedding_size and self.vector_size != embedding_size:
                raise RuntimeError(
                    f"Dimension mismatch Qdrant={self.vector_size} vs Embedding={embedding_size}. "
                    f"Recrée la collection '{collection_name}'."
                )

            logger.info(
                f"✓ Collection '{collection_name}' chargée — "
                f"{self.points_count} vecteurs, dim={self.vector_size}"
            )

            if self.points_count == 0:
                logger.warning(
                    f"La collection '{collection_name}' est VIDE. "
                    "Aucune réponse documentée ne sera possible. "
                    "Lancez le script d'ingestion pour indexer vos PDFs."
                )

        except Exception as e:
            raise RuntimeError(
                f"Impossible d'accéder à la collection Qdrant '{collection_name}' : {e}. "
                "Vérifiez QDRANT_URL, QDRANT_API_KEY et le nom de la collection."
            )

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        try:
            query_embedding = self.embedding_function.embed_query(query)
        except Exception as e:
            logger.error(f"Erreur génération embedding requête : {e}")
            return []

        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=k,
                with_payload=True,
            )
            hits = result.points
        except Exception as e:
            logger.error(f"Erreur recherche Qdrant : {e}")
            return []

        results = []
        for point in hits:
            payload = point.payload or {}

            page_content = _extraire_texte_payload(payload)
            metadata = _extraire_metadata_payload(payload)

            distance = 1.0 - point.score

            doc = Document(page_content=page_content, metadata=metadata)
            results.append((doc, distance))

        return results

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]

    def as_retriever(self, **kwargs):
        self.search_kwargs = kwargs.get("search_kwargs", {})
        self.search_type = kwargs.get("search_type", "similarity_score_threshold")
        return self

    def invoke(self, input_val) -> List[Document]:
        query = (
            input_val.get("query")
            if isinstance(input_val, dict)
            else str(input_val)
        )
        k = self.search_kwargs.get("k", 4)
        return self.similarity_search(query, k)

    def get_relevant_documents(self, query: str) -> List[Document]:
        k = self.search_kwargs.get("k", 4)
        return self.similarity_search(query, k)
