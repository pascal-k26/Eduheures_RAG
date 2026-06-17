import re
import logging
from typing import Optional
from src.qdrant_wrapper import QdrantRetrieverWrapper
from src.config import NOMBRE_DOCS_RECUPERES, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

logger = logging.getLogger(__name__)

# Seuil de distance cosinus max (0 = identique, 1 = opposé)
SEUIL_DISTANCE_MAX = 0.7


def extraire_signal_recherche(reponse_brute: str) -> tuple[Optional[str], str]:
    """Détecte et extrait le signal [RECHERCHE: ...] de la réponse du LLM."""
    match = re.search(r"\[RECHERCHE:\s*([^\]]+)\]", reponse_brute, re.IGNORECASE)
    if match:
        query = match.group(1).strip()
        texte_restant = reponse_brute[: match.start()] + reponse_brute[match.end():]
        return query, texte_restant.strip()
    return None, reponse_brute


def recuperer_documents(query: str, base: QdrantRetrieverWrapper) -> list:
    """Recherche vectorielle Qdrant avec filtre de pertinence."""
    resultats = base.similarity_search_with_score(query, k=NOMBRE_DOCS_RECUPERES)
    resultats_filtres = [
        (doc, score) for doc, score in resultats if score <= SEUIL_DISTANCE_MAX
    ]
    if not resultats_filtres:
        logger.info(f"Aucun résultat Qdrant sous le seuil {SEUIL_DISTANCE_MAX} pour : '{query[:60]}'")
        return resultats[:1] if resultats else []
    return resultats_filtres


def charger_base_qdrant() -> QdrantRetrieverWrapper:
    """Charge le wrapper Qdrant connecté à la collection UVCI."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise EnvironmentError(
            "QDRANT_URL et QDRANT_API_KEY sont requis. Vérifie ton .env"
        )
    logger.info(f"Connexion Qdrant Cloud — collection '{QDRANT_COLLECTION}'")
    return QdrantRetrieverWrapper(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION,
    )
