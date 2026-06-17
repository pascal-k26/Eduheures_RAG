import os
import json
import time
import logging
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import (
    DOSSIER_DONNEES_BRUTES,
    DOSSIER_DONNEES_TRAITEES,
    TAILLE_CHUNK,
    CHEVAUCHEMENT_CHUNK,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
)
from src.utils import creer_modele_embedding

logger = logging.getLogger(__name__)


# Nettoyage du texte

def nettoyer_texte(texte: str) -> str:
    import re
    texte = re.sub(r"[\x80-\xff]{3,}", "", texte)
    texte = re.sub(r" {2,}", " ", texte)
    texte = re.sub(r"\n\n\n+", "\n\n", texte)
    lines = [l.strip() for l in texte.split("\n") if l.strip() and not re.match(r"^[.\-]+$", l.strip())]
    return "\n".join(lines).strip()


# Chargement des documents

def charger_documents(dossier: str) -> list:
    if not os.path.exists(dossier):
        raise FileNotFoundError(f"Dossier introuvable : {dossier}")

    documents = []
    fichiers_pdf = [f for f in os.listdir(dossier) if f.endswith(".pdf")]

    if not fichiers_pdf:
        raise FileNotFoundError(f"Aucun fichier PDF trouvé dans {dossier}.")

    print(f"  {len(fichiers_pdf)} fichier(s) PDF trouvé(s)")

    for i, fichier in enumerate(fichiers_pdf, 1):
        chemin = os.path.join(dossier, fichier)
        try:
            print(f"  [{i}/{len(fichiers_pdf)}] Chargement : {fichier}...", end=" ")
            loader = PyPDFLoader(chemin)
            docs = loader.load()
            for doc in docs:
                doc.page_content = nettoyer_texte(doc.page_content)
                # Ajouter métadonnées UVCI
                doc.metadata["universite"] = "UVCI"
                doc.metadata["categorie"]  = _detecter_categorie(fichier)
            documents.extend(docs)
            print(f"✓ ({len(docs)} pages)")
        except Exception as e:
            print(f"ERREUR : {e}")
            continue

    if not documents:
        raise FileNotFoundError("Aucun document valide chargé.")

    print(f"\n  Total : {len(documents)} page(s) chargée(s).")
    return documents


def _detecter_categorie(nom_fichier: str) -> str:
    """Détecte la catégorie du document UVCI à partir du nom de fichier."""
    nom = nom_fichier.lower()
    if any(k in nom for k in ["inscription", "admission", "candidature"]):
        return "inscription"
    if any(k in nom for k in ["programme", "formation", "licence", "master", "bts"]):
        return "formation"
    if any(k in nom for k in ["calendrier", "planning", "examen"]):
        return "calendrier"
    if any(k in nom for k in ["frais", "scolarite", "tarif", "bourse"]):
        return "frais"
    if any(k in nom for k in ["contact", "service", "administration"]):
        return "contact"
    return "general"


# Découpage

def decouper_documents(documents: list) -> list:
    decoupeur = RecursiveCharacterTextSplitter(
        chunk_size=TAILLE_CHUNK,
        chunk_overlap=CHEVAUCHEMENT_CHUNK,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    morceaux = decoupeur.split_documents(documents)
    print(f"  {len(morceaux)} chunk(s) créé(s).")
    return morceaux


# Ingestion Qdrant

def ingerer_dans_qdrant(morceaux: list) -> None:
    """Indexe les chunks dans Qdrant Cloud."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise EnvironmentError("QDRANT_URL et QDRANT_API_KEY sont requis dans .env")

    modele_embedding = creer_modele_embedding()
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    # Vérification de la dimension d'embedding
    test_vec = modele_embedding.embed_query("test")
    dim = len(test_vec)
    print(f"\n  Dimension d'embedding : {dim}")

    # Création ou vérification de la collection
    collections_existantes = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections_existantes:
        print(f"  Création de la collection '{QDRANT_COLLECTION}'...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"  Collection créée.")
    else:
        print(f"  Collection '{QDRANT_COLLECTION}' existante — ajout des documents.")

    # Indexation par batch
    taille_batch = 10
    total_batches = (len(morceaux) + taille_batch - 1) // taille_batch
    print(f"\n  Indexation de {len(morceaux)} chunks en {total_batches} batch(es)...")

    for i in range(0, len(morceaux), taille_batch):
        batch = morceaux[i : i + taille_batch]
        batch_num = i // taille_batch + 1

        textes = [m.page_content for m in batch]
        vecteurs = modele_embedding.embed_documents(textes)

        points = [
            PointStruct(
                id=i + j,
                vector=vecteur,
                payload={
                    "page_content": batch[j].page_content,
                    "metadata": {
                        "source":     batch[j].metadata.get("source", "Inconnu"),
                        "page":       batch[j].metadata.get("page", 0),
                        "universite": batch[j].metadata.get("universite", "UVCI"),
                        "categorie":  batch[j].metadata.get("categorie", "general"),
                    },
                },
            )
            for j, vecteur in enumerate(vecteurs)
        ]

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"  [{batch_num}/{total_batches}] {len(points)} point(s) indexé(s)")
        time.sleep(0.2)  # Respect du rate limit Qdrant Cloud

    info = client.get_collection(QDRANT_COLLECTION)
    print(f"\n  ✓ Indexation terminée — {info.points_count} vecteurs dans '{QDRANT_COLLECTION}'")


#  Point d'entrée 
def executer_ingestion():
    """Lance l'ingestion complète des documents UVCI vers Qdrant."""
    print("=" * 60)
    print("  EDUHEURES — Ingestion des documents UVCI vers Qdrant")
    print("=" * 60)

    print(f"\n[1/3] Chargement des PDF depuis '{DOSSIER_DONNEES_BRUTES}'...")
    documents = charger_documents(DOSSIER_DONNEES_BRUTES)

    print(f"\n[2/3] Découpage des documents...")
    morceaux = decouper_documents(documents)

    print(f"\n[3/3] Indexation dans Qdrant Cloud...")
    ingerer_dans_qdrant(morceaux)

    print("\n✓ Ingestion terminée avec succès !")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    executer_ingestion()
