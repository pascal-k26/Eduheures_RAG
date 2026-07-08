import os
from dotenv import load_dotenv
load_dotenv()

# azure openai
AZURE_CHAT_API_KEY      = os.getenv("OPENAI_API_KEY")
AZURE_CHAT_ENDPOINT     = os.getenv("OPENAI_API_ENDPOINT")
AZURE_CHAT_DEPLOYMENT   = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini")
AZURE_CHAT_API_VERSION  = os.getenv("AZURE_CHAT_API_VERSION", "2024-10-21")
# utilise uniquement pour les modeles reasoning (gpt-5*, o1, o3, o4*)
AZURE_CHAT_REASONING_EFFORT = os.getenv("AZURE_CHAT_REASONING_EFFORT", "medium")

# embeddings azure openai
AZURE_EMBEDDING_API_KEY     = os.getenv("OPENAI_EMBEDDING_API_KEY")
AZURE_EMBEDDING_ENDPOINT    = os.getenv("OPENAI_EMBEDDING_API_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT  = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15")

# qdrant cloud
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "uvci_documents")

# nom des vecteurs nommes dans qdrant, dense et sparse
NOM_VECTEUR_DENSE  = "dense"
NOM_VECTEUR_SPARSE = "sparse"
MODELE_SPARSE      = os.getenv("MODELE_SPARSE", "Qdrant/bm25")

# recherche web agentique
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# sites uvci prioritaires pour la recherche web
UVCI_SITES = [
    "scolarite.uvci.edu.ci",
    "biblio.uvci.edu.ci",
    "mesrs.ci",
    "espacenumerique.uvci.edu.ci",
    "uvci.online",
    "campus.uvci.online",
    "uvci.tv",
]
UVCI_SEARCH_CONTEXT = "UVCI universite virtuelle Cote d'Ivoire"

# chemins
DOSSIER_DONNEES_BRUTES   = "data/raw"
DOSSIER_DONNEES_TRAITEES = "data/processed"
FICHIER_PROMPT           = "prompts/rag_prompt.txt"

# decoupage, en tokens tiktoken cl100k_base, pas en caracteres
TAILLE_CHUNK_TOKENS        = int(os.getenv("TAILLE_CHUNK_TOKENS", 400))
CHEVAUCHEMENT_CHUNK_TOKENS = int(os.getenv("CHEVAUCHEMENT_CHUNK_TOKENS", 50))

# retrieval
NOMBRE_DOCS_RECUPERES = 5
SEUIL_DISTANCE_MAX    = float(os.getenv("SEUIL_DISTANCE_MAX", 0.7))

# agent
MAX_TOURS_AGENT = int(os.getenv("MAX_TOURS_AGENT", 4))
MAX_HISTORIQUE  = 10