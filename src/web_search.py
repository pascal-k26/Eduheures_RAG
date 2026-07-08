import logging
import requests
from src.config import TAVILY_API_KEY, SERPER_API_KEY, UVCI_SITES, UVCI_SEARCH_CONTEXT

logger = logging.getLogger(__name__)


class ResultatWeb:
    def __init__(self, titre: str, url: str, contenu: str, source: str = "web"):
        self.titre = titre
        self.url = url
        self.contenu = contenu
        self.source = source

    def vers_texte(self) -> str:
        return f"{self.titre}\nsource: {self.url}\n{self.contenu}"


def _rechercher_tavily(query: str, nb_resultats: int, domaines: list = None) -> list:
    if not TAVILY_API_KEY:
        return []
    try:
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "max_results": nb_resultats,
            "include_answer": True,
        }
        if domaines:
            payload["include_domains"] = domaines
        response = requests.post("https://api.tavily.com/search", json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        return [
            ResultatWeb(r.get("title", "sans titre"), r.get("url", ""), r.get("content", ""), "tavily")
            for r in data.get("results", [])
        ]
    except Exception as e:
        logger.warning(f"tavily echoue: {e}")
        return []


def _rechercher_serper(query: str, nb_resultats: int, domaines: list = None) -> list:
    if not SERPER_API_KEY:
        return []
    try:
        requete = query
        if domaines:
            sites = " OR ".join(f"site:{d}" for d in domaines)
            requete = f"{query} ({sites})"
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": requete, "num": nb_resultats, "hl": "fr", "gl": "ci"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return [
            ResultatWeb(r.get("title", "sans titre"), r.get("link", ""), r.get("snippet", ""), "serper")
            for r in data.get("organic", [])
        ]
    except Exception as e:
        logger.warning(f"serper echoue: {e}")
        return []


def _rechercher_duckduckgo(query: str, nb_resultats: int) -> list:
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1", "skip_disambig": "1"},
            timeout=10,
            headers={"User-Agent": "Eduheures-UVCI/1.0"},
        )
        response.raise_for_status()
        data = response.json()
        resultats = []
        if data.get("AbstractText"):
            resultats.append(ResultatWeb(
                data.get("Heading", "resume"), data.get("AbstractURL", ""), data["AbstractText"], "duckduckgo"
            ))
        for topic in data.get("RelatedTopics", [])[:nb_resultats - len(resultats)]:
            if isinstance(topic, dict) and topic.get("Text"):
                resultats.append(ResultatWeb(topic.get("Text", "")[:80], topic.get("FirstURL", ""), topic.get("Text", ""), "duckduckgo"))
        return resultats
    except Exception as e:
        logger.warning(f"duckduckgo echoue: {e}")
        return []


def rechercher_sur_le_web(query: str, nb_resultats: int = 5, cibler_uvci: bool = True) -> list:
    domaines = UVCI_SITES if cibler_uvci else None
    query_complete = f"{query} {UVCI_SEARCH_CONTEXT}" if cibler_uvci else query

    resultats = _rechercher_tavily(query_complete, nb_resultats, domaines)
    if resultats:
        return resultats

    resultats = _rechercher_serper(query_complete, nb_resultats, domaines)
    if resultats:
        return resultats

    return _rechercher_duckduckgo(query_complete, nb_resultats)


def formater_resultats_web(resultats: list) -> str:
    if not resultats:
        return "aucun resultat web trouve"
    return "\n\n".join(f"resultat {i}\n{r.vers_texte()}" for i, r in enumerate(resultats, 1))