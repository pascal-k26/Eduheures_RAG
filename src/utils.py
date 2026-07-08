import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.config import (
    AZURE_CHAT_API_KEY, AZURE_CHAT_ENDPOINT, AZURE_CHAT_DEPLOYMENT, AZURE_CHAT_API_VERSION,
    AZURE_CHAT_REASONING_EFFORT,
    AZURE_EMBEDDING_API_KEY, AZURE_EMBEDDING_ENDPOINT, AZURE_EMBEDDING_DEPLOYMENT, AZURE_EMBEDDING_API_VERSION,
)

_moteur_chat = None
_moteur_embedding = None

# prefixes des modeles reasoning azure openai
_PREFIXES_MODELES_REASONING = ("gpt-5", "o1", "o3", "o4")


def _est_modele_reasoning(nom_deploiement: str) -> bool:
    nom = (nom_deploiement or "").lower()
    return any(nom.startswith(prefixe) for prefixe in _PREFIXES_MODELES_REASONING)


def creer_modele_chat():
    global _moteur_chat
    if _moteur_chat is None:
        if not AZURE_CHAT_API_KEY or not AZURE_CHAT_ENDPOINT:
            raise ValueError("credentials azure openai chat manquantes, verifie ton .env")

        kwargs = {
            "azure_endpoint": AZURE_CHAT_ENDPOINT,
            "azure_deployment": AZURE_CHAT_DEPLOYMENT,
            "openai_api_key": AZURE_CHAT_API_KEY,
            "openai_api_version": AZURE_CHAT_API_VERSION,
        }

        if _est_modele_reasoning(AZURE_CHAT_DEPLOYMENT):
            # gpt-5*, o1, o3, o4* : temperature interdite (sauf valeur par defaut 1
            kwargs["reasoning_effort"] = AZURE_CHAT_REASONING_EFFORT
        else:
            temperature = os.getenv("AZURE_CHAT_TEMPERATURE")
            if temperature is not None and temperature != "":
                kwargs["temperature"] = float(temperature)

        _moteur_chat = AzureChatOpenAI(**kwargs)
    return _moteur_chat


def creer_modele_embedding():
    global _moteur_embedding
    if _moteur_embedding is None:
        if not AZURE_EMBEDDING_API_KEY or not AZURE_EMBEDDING_ENDPOINT:
            raise ValueError("credentials azure openai embedding manquantes, verifie ton .env")
        _moteur_embedding = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
            openai_api_key=AZURE_EMBEDDING_API_KEY,
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            api_version=AZURE_EMBEDDING_API_VERSION,
        )
    return _moteur_embedding


def lire_prompt(chemin_fichier: str) -> str:
    if not os.path.exists(chemin_fichier):
        raise FileNotFoundError(f"fichier prompt introuvable: {chemin_fichier}")
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        return f.read()


def formater_documents(documents: list) -> str:
    # formate les documents qdrant en texte structure pour le prompt
    if not documents:
        return "aucun document pertinent trouve"

    parties = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "source inconnue")
        nom_fichier = os.path.basename(source)
        page = doc.metadata.get("page")
        categorie = doc.metadata.get("categorie", "")
        date_ingestion = doc.metadata.get("date_ingestion", "")

        ref = nom_fichier
        if page is not None:
            ref += f", page {int(page) + 1}"
        if categorie:
            ref += f" [{categorie}]"
        if date_ingestion:
            ref += f" [indexe le {date_ingestion}]"

        parties.append(f"[Document {i}, {ref}]\n{doc.page_content}")

    return "\n\n".join(parties)
