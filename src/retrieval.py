import logging
from src.qdrant_wrapper import QdrantRetrieverWrapper
from src.config import NOMBRE_DOCS_RECUPERES, SEUIL_DISTANCE_MAX, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

logger = logging.getLogger(__name__)


def recuperer_documents(query: str, base: QdrantRetrieverWrapper) -> list:
    # recherche hybride dense + sparse, fusion rrf faite dans le wrapper
    resultats = base.similarity_search_with_score(query, k=NOMBRE_DOCS_RECUPERES)
    resultats_filtres = [(doc, score) for doc, score in resultats if score <= SEUIL_DISTANCE_MAX]
    if not resultats_filtres:
        return resultats[:1] if resultats else []
    return resultats_filtres


def charger_base_qdrant() -> QdrantRetrieverWrapper:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise EnvironmentError("QDRANT_URL et QDRANT_API_KEY requis dans .env")
    logger.info(f"connexion qdrant, collection '{QDRANT_COLLECTION}'")
    return QdrantRetrieverWrapper(url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=QDRANT_COLLECTION)