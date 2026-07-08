import os
import re
import time
import uuid
import hashlib
import logging
from datetime import date

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, PayloadSchemaType,
)
import tiktoken

from src.config import (
    DOSSIER_DONNEES_BRUTES, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    NOM_VECTEUR_DENSE, NOM_VECTEUR_SPARSE,
    TAILLE_CHUNK_TOKENS, CHEVAUCHEMENT_CHUNK_TOKENS,
)
from src.utils import creer_modele_embedding
from src.qdrant_wrapper import creer_modele_sparse

logger = logging.getLogger(__name__)

_encodeur = tiktoken.get_encoding("cl100k_base")


def _compter_tokens(texte: str) -> int:
    return len(_encodeur.encode(texte))


def nettoyer_texte(texte: str) -> str:
    texte = re.sub(r"[\x80-\xff]{3,}", "", texte)
    texte = re.sub(r" {2,}", " ", texte)
    texte = re.sub(r"\n\n\n+", "\n\n", texte)
    lignes = [l.strip() for l in texte.split("\n") if l.strip() and not re.match(r"^[.\-]+$", l.strip())]
    return "\n".join(lignes).strip()


def _detecter_categorie(nom_fichier: str, texte: str) -> str:
    # regarde le nom du fichier et le debut du contenu, pas seulement le nom
    cible = (nom_fichier + " " + texte[:300]).lower()
    if any(k in cible for k in ["inscription", "admission", "candidature"]):
        return "inscription"
    if any(k in cible for k in ["programme", "formation", "licence", "master", "bts"]):
        return "formation"
    if any(k in cible for k in ["calendrier", "planning", "examen"]):
        return "calendrier"
    if any(k in cible for k in ["frais", "scolarite", "tarif", "bourse"]):
        return "frais"
    if any(k in cible for k in ["contact", "service", "administration"]):
        return "contact"
    return "general"


def charger_documents(dossier: str) -> list:
    if not os.path.exists(dossier):
        raise FileNotFoundError(f"dossier introuvable: {dossier}")

    fichiers_pdf = [f for f in os.listdir(dossier) if f.endswith(".pdf")]
    if not fichiers_pdf:
        raise FileNotFoundError(f"aucun pdf trouve dans {dossier}")

    documents = []
    for fichier in fichiers_pdf:
        chemin = os.path.join(dossier, fichier)
        try:
            docs = PyPDFLoader(chemin).load()
            for doc in docs:
                doc.page_content = nettoyer_texte(doc.page_content)
                doc.metadata["categorie"] = _detecter_categorie(fichier, doc.page_content)
            documents.extend(docs)
            print(f"charge: {fichier} ({len(docs)} pages)")
        except Exception as e:
            print(f"erreur sur {fichier}: {e}")

    if not documents:
        raise FileNotFoundError("aucun document valide charge")
    return documents


def decouper_documents(documents: list) -> list:
    decoupeur = RecursiveCharacterTextSplitter(
        chunk_size=TAILLE_CHUNK_TOKENS,
        chunk_overlap=CHEVAUCHEMENT_CHUNK_TOKENS,
        length_function=_compter_tokens,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    morceaux = decoupeur.split_documents(documents)

    # numerote les chunks dans l'ordre par fichier source
    compteurs = {}
    for m in morceaux:
        source = m.metadata.get("source", "inconnu")
        compteurs[source] = compteurs.get(source, -1) + 1
        m.metadata["chunk_index"] = compteurs[source]

    print(f"{len(morceaux)} chunks crees")
    return morceaux


def _assurer_collection(client: QdrantClient, dim: int):
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={NOM_VECTEUR_DENSE: VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={NOM_VECTEUR_SPARSE: SparseVectorParams()},
        )
        print(f"collection '{QDRANT_COLLECTION}' creee")

    client.create_payload_index(QDRANT_COLLECTION, "metadata.source", PayloadSchemaType.KEYWORD)
    client.create_payload_index(QDRANT_COLLECTION, "metadata.categorie", PayloadSchemaType.KEYWORD)


def _supprimer_anciens_points(client: QdrantClient, sources: set):
    # supprime les points existants d'un fichier avant de le reindexer
    for source in sources:
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]),
        )


def ingerer_dans_qdrant(morceaux: list) -> None:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise EnvironmentError("QDRANT_URL et QDRANT_API_KEY requis dans .env")

    modele_dense = creer_modele_embedding()
    modele_sparse = creer_modele_sparse()
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    dim = len(modele_dense.embed_query("test"))
    _assurer_collection(client, dim)

    sources = {m.metadata.get("source", "inconnu") for m in morceaux}
    _supprimer_anciens_points(client, sources)

    aujourdhui = date.today().isoformat()
    taille_batch = 10
    total_batches = (len(morceaux) + taille_batch - 1) // taille_batch

    for i in range(0, len(morceaux), taille_batch):
        batch = morceaux[i:i + taille_batch]
        textes = [m.page_content for m in batch]

        vecteurs_denses = modele_dense.embed_documents(textes)
        vecteurs_sparses = list(modele_sparse.embed(textes))

        points = []
        for m, v_dense, v_sparse in zip(batch, vecteurs_denses, vecteurs_sparses):
            hash_contenu = hashlib.sha256(m.page_content.encode("utf-8")).hexdigest()[:16]
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    NOM_VECTEUR_DENSE: v_dense,
                    NOM_VECTEUR_SPARSE: {
                        "indices": v_sparse.indices.tolist(),
                        "values": v_sparse.values.tolist(),
                    },
                },
                payload={
                    "page_content": m.page_content,
                    "metadata": {
                        "source": m.metadata.get("source", "inconnu"),
                        "page": m.metadata.get("page", 0),
                        "categorie": m.metadata.get("categorie", "general"),
                        "chunk_index": m.metadata.get("chunk_index", 0),
                        "date_ingestion": aujourdhui,
                        "hash_contenu": hash_contenu,
                    },
                },
            ))

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"batch {i // taille_batch + 1}/{total_batches} indexe")
        time.sleep(0.2)

    info = client.get_collection(QDRANT_COLLECTION)
    print(f"indexation terminee, {info.points_count} points dans '{QDRANT_COLLECTION}'")


def executer_ingestion():
    print("ingestion uvci vers qdrant")
    documents = charger_documents(DOSSIER_DONNEES_BRUTES)
    morceaux = decouper_documents(documents)
    ingerer_dans_qdrant(morceaux)
    print("ingestion terminee")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    executer_ingestion()