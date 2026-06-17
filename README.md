# Eduheures — Assistant Universitaire UVCI

Assistant RAG pour l'Université Virtuelle de Côte d'Ivoire. Répond aux questions sur les formations, inscriptions, frais, calendriers et actualités en combinant une base documentaire interne (Qdrant) et une recherche web agentique.

## Architecture

Le système fonctionne avec une orchestration RAG à deux passes :

1. **Première passe** : Le LLM décide des besoins de recherche (documents internes et/ou web)
2. **Deuxième passe** : Le LLM génère la réponse enrichie avec sources

Composants : Azure OpenAI (LLM + embeddings), Qdrant Cloud (documents UVCI), Recherche web (Tavily/Serper/DuckDuckGo), FastAPI (API REST)

## Structure du projet

```
.
├── app.py                 CLI
├── server_fastapi.py      API FastAPI
├── config.yaml            Configuration
├── requirements.txt       Dépendances
├── src/
│   ├── config.py          Variables d'environnement
│   ├── generation.py      Orchestration RAG
│   ├── retrieval.py       Recherche vectorielle Qdrant
│   ├── web_search.py      Recherche web
│   ├── ingestion.py       Pipeline PDF → Qdrant
│   ├── qdrant_wrapper.py  Client Qdrant
│   └── utils.py           Utilitaires
├── prompts/
│   └── rag_prompt.txt     Prompt système
└── data/raw/              Documents PDF à indexer
```

## Installation

```bash
git clone <votre-repo>
cd eduheures
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Créez un fichier `.env` :

```env
OPENAI_API_KEY=<clé Azure>
OPENAI_API_ENDPOINT=<endpoint>
OPENAI_EMBEDDING_API_KEY=<clé embedding>
OPENAI_EMBEDDING_API_ENDPOINT=<endpoint embedding>
QDRANT_URL=<URL Qdrant Cloud>
QDRANT_API_KEY=<clé API Qdrant>
TAVILY_API_KEY=<optionnel>
SERPER_API_KEY=<optionnel>
```

Voir [config.py](src/config.py) pour toutes les variables.

## Démarrage

Mode CLI :

```bash
python app.py
```

Mode API :

```bash
uvicorn server_fastapi:app --reload --port 8000
```

Documentation Swagger : http://localhost:8000/docs

## Ingestion des documents

```bash
# Placer vos PDF dans data/raw/
python -m src.ingestion
```

Indexe automatiquement avec détection de catégorie : inscription, formation, calendrier, frais, contact, general

## Endpoints API

| Méthode | Endpoint            | Description       |
| ------- | ------------------- | ----------------- |
| GET     | `/health`           | Statut API        |
| POST    | `/session/nouvelle` | Créer session     |
| DELETE  | `/session/{id}`     | Supprimer session |
| POST    | `/chat`             | Envoyer message   |

Exemple :

```json
POST /chat
{
  "session_id": "uuid",
  "message": "Conditions d'inscription ?"
}
```

Réponse :

```json
{
  "reponse": "Pour s'inscrire...",
  "sources": [
    { "fichier": "guide_inscription.pdf", "type": "document" },
    { "fichier": "https://uvci.edu.ci", "type": "web" }
  ]
}
```

## Recherche web

L'assistant déclenche une recherche web si :

- Question avec mots-clés d'actualité (récent, nouveau, 2025/2026, événement)
- Base Qdrant insuffisante
- Signal `[RECHERCHE_WEB: ...]` généré par le LLM

Domaines prioritaires UVCI : scolarite.uvci.edu.ci, biblio.uvci.edu.ci, campus.uvci.online, mesrs.ci

## Paramètres clés

| Paramètre             | Valeur | Rôle                                 |
| --------------------- | ------ | ------------------------------------ |
| TAILLE_CHUNK          | 1000   | Taille des segments texte            |
| NOMBRE_DOCS_RECUPERES | 5      | Documents récupérés par requête      |
| SEUIL_DISTANCE_MAX    | 0.7    | Filtre pertinence (distance cosinus) |
| temperature           | 0.5    | Déterminisme réponse LLM             |
| MAX_HISTORIQUE        | 10     | Échanges conservés                   |

## Dépendances

- **LLM/RAG** : LangChain, OpenAI, Qdrant
- **API** : FastAPI, Uvicorn, pydantic
- **Web** : Tavily, Serper, requests
- **PDF** : PyPDF
- **Sécurité** : python-dotenv, slowapi

Voir [requirements.txt](requirements.txt) pour les versions.

## Troubleshooting

| Problème                | Solution                                        |
| ----------------------- | ----------------------------------------------- |
| Erreur Qdrant           | Vérifier `.env` et connexion                    |
| Aucun document indexé   | Vérifier `data/raw/` et permissions             |
| Pas de sources trouvées | Augmenter NOMBRE_DOCS_RECUPERES ou ajouter docs |
| Erreur Azure OpenAI     | Vérifier clés API et région                     |

## Licence

À définir selon vos besoins
