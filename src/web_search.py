"""
Module de recherche web agentique pour Eduheures / UVCI.

Stratégie :
  1. Tavily Search API  (si TAVILY_API_KEY défini)  — résultats riches et filtrables
  2. Serper API         (si SERPER_API_KEY défini)   — alternative Google Search
  3. DuckDuckGo         (fallback, sans clé)          — toujours disponible

Les recherches sont volontairement ciblées sur les sites officiels de l'UVCI
avant d'élargir à tout le web si nécessaire.
"""

import logging
import os
import re
import requests
from typing import Optional
from src.config import TAVILY_API_KEY, SERPER_API_KEY, UVCI_SITES, UVCI_SEARCH_CONTEXT

logger = logging.getLogger(__name__)


# ── Structures de résultat ────────────────────────────────────────────────────

class ResultatWeb:
    """Un résultat de recherche web normalisé."""

    def __init__(self, titre: str, url: str, contenu: str, source: str = "web"):
        self.titre   = titre
        self.url     = url
        self.contenu = contenu
        self.source  = source  # "tavily" | "serper" | "duckduckgo"

    def __repr__(self):
        return f"<ResultatWeb titre='{self.titre[:50]}' url='{self.url}'>"

    def vers_texte(self) -> str:
        """Formate le résultat pour l'injection dans le prompt LLM."""
        return (
            f"**{self.titre}**\n"
            f"Source : {self.url}\n"
            f"{self.contenu}"
        )


# ── Moteur Tavily ─────────────────────────────────────────────────────────────

def _rechercher_tavily(query: str, nb_resultats: int = 5) -> list[ResultatWeb]:
    """Recherche via Tavily Search API — meilleure option pour le RAG."""
    if not TAVILY_API_KEY:
        return []

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": nb_resultats,
                "include_raw_content": False,
                "include_answer": True,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        resultats = []
        for r in data.get("results", []):
            resultats.append(ResultatWeb(
                titre   = r.get("title", "Sans titre"),
                url     = r.get("url", ""),
                contenu = r.get("content", ""),
                source  = "tavily",
            ))
        logger.info(f"Tavily : {len(resultats)} résultat(s) pour '{query[:60]}'")
        return resultats

    except Exception as e:
        logger.warning(f"Tavily échoué : {e}")
        return []


# Moteur Serper

def _rechercher_serper(query: str, nb_resultats: int = 5) -> list[ResultatWeb]:
    """Recherche via Serper.dev (Google Search API)."""
    if not SERPER_API_KEY:
        return []

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": nb_resultats, "hl": "fr", "gl": "ci"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        resultats = []
        for r in data.get("organic", []):
            resultats.append(ResultatWeb(
                titre   = r.get("title", "Sans titre"),
                url     = r.get("link", ""),
                contenu = r.get("snippet", ""),
                source  = "serper",
            ))
        logger.info(f"Serper : {len(resultats)} résultat(s) pour '{query[:60]}'")
        return resultats

    except Exception as e:
        logger.warning(f"Serper échoué : {e}")
        return []


# Moteur DuckDuckGo (fallback sans clé) 

def _rechercher_duckduckgo(query: str, nb_resultats: int = 5) -> list[ResultatWeb]:
    """Recherche via l'API instantAnswer DuckDuckGo (fallback gratuit)."""
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1",
            },
            timeout=10,
            headers={"User-Agent": "Eduheures-UVCI-Assistant/1.0"},
        )
        response.raise_for_status()
        data = response.json()

        resultats = []

        # AbstractText — résumé principal
        if data.get("AbstractText"):
            resultats.append(ResultatWeb(
                titre   = data.get("Heading", "Résumé"),
                url     = data.get("AbstractURL", "https://duckduckgo.com"),
                contenu = data["AbstractText"],
                source  = "duckduckgo",
            ))

        # RelatedTopics — topics connexes
        for topic in data.get("RelatedTopics", [])[:nb_resultats - len(resultats)]:
            if isinstance(topic, dict) and topic.get("Text"):
                resultats.append(ResultatWeb(
                    titre   = topic.get("Text", "")[:80],
                    url     = topic.get("FirstURL", ""),
                    contenu = topic.get("Text", ""),
                    source  = "duckduckgo",
                ))

        logger.info(f"DuckDuckGo : {len(resultats)} résultat(s) pour '{query[:60]}'")
        return resultats

    except Exception as e:
        logger.warning(f"DuckDuckGo échoué : {e}")
        return []


# Interface publique

def rechercher_sur_le_web(
    query: str,
    nb_resultats: int = 5,
    cibler_uvci: bool = True,
) -> list[ResultatWeb]:
    """
    Recherche web agentique — essaie Tavily, puis Serper, puis DuckDuckGo.

    Si cibler_uvci=True, préfixe la requête avec les sites UVCI
    pour maximiser la pertinence avant d'élargir si aucun résultat UVCI.
    """
    query_complete = f"{query} {UVCI_SEARCH_CONTEXT}" if cibler_uvci else query

    # Tentative 1 : Tavily
    resultats = _rechercher_tavily(query_complete, nb_resultats)
    if resultats:
        return resultats

    # Tentative 2 : Serper
    resultats = _rechercher_serper(query_complete, nb_resultats)
    if resultats:
        return resultats

    # Tentative 3 : DuckDuckGo (fallback)
    return _rechercher_duckduckgo(query_complete, nb_resultats)


def formater_resultats_web(resultats: list[ResultatWeb]) -> str:
    """Formate une liste de résultats web en texte pour le prompt LLM."""
    if not resultats:
        return "Aucun résultat web trouvé."

    parties = []
    for i, r in enumerate(resultats, 1):
        parties.append(f"[Résultat web {i}]\n{r.vers_texte()}")

    return "\n\n".join(parties)


def extraire_signal_recherche_web(reponse_brute: str) -> tuple[Optional[str], str]:
    """Détecte et extrait le signal [RECHERCHE_WEB: ...] de la réponse du LLM."""
    match = re.search(r"\[RECHERCHE_WEB:\s*([^\]]+)\]", reponse_brute, re.IGNORECASE)
    if match:
        query = match.group(1).strip()
        texte_restant = reponse_brute[: match.start()] + reponse_brute[match.end():]
        return query, texte_restant.strip()
    return None, reponse_brute
