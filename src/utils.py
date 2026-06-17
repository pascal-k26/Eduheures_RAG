import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.config import (
    AZURE_CHAT_API_KEY, AZURE_CHAT_ENDPOINT, AZURE_CHAT_DEPLOYMENT, AZURE_CHAT_API_VERSION,
    AZURE_EMBEDDING_API_KEY, AZURE_EMBEDDING_ENDPOINT, AZURE_EMBEDDING_DEPLOYMENT, AZURE_EMBEDDING_API_VERSION,
)

_moteur_chat      = None
_moteur_embedding = None


def creer_modele_chat():
    global _moteur_chat
    if _moteur_chat is None:
        if not AZURE_CHAT_API_KEY or not AZURE_CHAT_ENDPOINT:
            raise ValueError("Credentials Azure OpenAI Chat manquantes. Vérifie ton .env")
        _moteur_chat = AzureChatOpenAI(
            azure_endpoint=AZURE_CHAT_ENDPOINT,
            azure_deployment=AZURE_CHAT_DEPLOYMENT,
            openai_api_key=AZURE_CHAT_API_KEY,
            openai_api_version=AZURE_CHAT_API_VERSION,
            temperature=0.5,
        )
    return _moteur_chat


def creer_modele_embedding():
    global _moteur_embedding
    if _moteur_embedding is None:
        if not AZURE_EMBEDDING_API_KEY or not AZURE_EMBEDDING_ENDPOINT:
            raise ValueError("Credentials Azure OpenAI Embedding manquantes. Vérifie ton .env")
        _moteur_embedding = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
            openai_api_key=AZURE_EMBEDDING_API_KEY,
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            api_version=AZURE_EMBEDDING_API_VERSION,
        )
    return _moteur_embedding


def lire_prompt(chemin_fichier: str) -> str:
    if not os.path.exists(chemin_fichier):
        raise FileNotFoundError(f"Fichier prompt introuvable : {chemin_fichier}")
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        return f.read()


def formater_documents(documents: list) -> str:
    """Formate les documents Qdrant en texte structuré pour le prompt."""
    if not documents:
        return "Aucun document pertinent trouvé."

    parties = []
    for i, doc in enumerate(documents, 1):
        source     = doc.metadata.get("source", "source inconnue")
        nom_fichier = os.path.basename(source)
        page       = doc.metadata.get("page")
        categorie  = doc.metadata.get("categorie", "")

        ref = f"{nom_fichier}"
        if page is not None:
            ref += f", page {int(page) + 1}"
        if categorie:
            ref += f" [{categorie}]"

        parties.append(f"[Document {i} — {ref}]\n{doc.page_content}")

    return "\n\n".join(parties)
