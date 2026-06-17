import os
from dotenv import load_dotenv
load_dotenv()

# LLM : Azure OpenAI Chat
AZURE_CHAT_API_KEY      = os.getenv("OPENAI_API_KEY")
AZURE_CHAT_ENDPOINT     = os.getenv("OPENAI_API_ENDPOINT")
AZURE_CHAT_DEPLOYMENT   = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4.1-mini")
AZURE_CHAT_API_VERSION  = os.getenv("AZURE_CHAT_API_VERSION", "2024-10-21")

# Embeddings : Azure OpenAI
AZURE_EMBEDDING_API_KEY     = os.getenv("OPENAI_EMBEDDING_API_KEY")
AZURE_EMBEDDING_ENDPOINT    = os.getenv("OPENAI_EMBEDDING_API_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT  = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15")

# Qdrant Cloud 
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "uvci_documents")

#  Recherche web agentique
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")          
SERPER_API_KEY = os.getenv("SERPER_API_KEY")          

# Sites UVCI prioritaires pour la recherche web
UVCI_SITES = [
    "scolarite.uvci.edu.ci",
    "biblio.uvci.edu.ci",
    "mesrs.ci",
    "espacenumerique.uvci.edu.ci", 
    "uvci.online", 
    "campus.uvci.online", 
    "uvci.tv"
]
UVCI_SEARCH_CONTEXT = "UVCI université virtuelle Côte d'Ivoire"

# Chemins 
DOSSIER_DONNEES_BRUTES   = "data/raw"
DOSSIER_DONNEES_TRAITEES = "data/processed"
DOSSIER_BASE_VECTEURS    = "vector_db"
FICHIER_PROMPT           = "prompts/rag_prompt.txt"

#  Découpage
TAILLE_CHUNK       = 1000
CHEVAUCHEMENT_CHUNK = 150

#  Retrieval 
NOMBRE_DOCS_RECUPERES = 5

# Génération
MAX_HISTORIQUE = 10
