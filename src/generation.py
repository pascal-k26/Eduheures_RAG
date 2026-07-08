import os
import logging
import dotenv
dotenv.load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.config import FICHIER_PROMPT, MAX_TOURS_AGENT
from src.utils import creer_modele_chat, lire_prompt, formater_documents
from src.retrieval import recuperer_documents, charger_base_qdrant
from src.web_search import rechercher_sur_le_web, formater_resultats_web

logger = logging.getLogger(__name__)

_prompt_cache = None
_base_cache = None


def _get_prompt() -> str:
    global _prompt_cache
    if _prompt_cache is None:
        try:
            _prompt_cache = lire_prompt(FICHIER_PROMPT)
        except FileNotFoundError:
            _prompt_cache = _prompt_par_defaut()
    return _prompt_cache


def _get_base():
    global _base_cache
    if _base_cache is None:
        _base_cache = charger_base_qdrant()
    return _base_cache


def repondre(question: str, historique: list = None) -> dict:
    historique = historique or []
    sources_collectees = []

    @tool
    def recherche_documents_internes(requete: str) -> str:
        """Recherche dans la base documentaire interne UVCI: formations, inscriptions, frais, calendriers, contacts."""
        try:
            base = _get_base()
        except Exception as e:
            return f"base documentaire indisponible: {e}"
        resultats = recuperer_documents(requete, base)
        if not resultats:
            return "aucun document interne pertinent trouve"
        for doc, distance in resultats:
            sources_collectees.append({
                "fichier": doc.metadata.get("source", "inconnu"),
                "nom_court": os.path.basename(doc.metadata.get("source", "inconnu")),
                "page": doc.metadata.get("page", "n/a"),
                "score": round(max(0.0, min(1.0, 1 - distance)) * 100, 1),
                "type": "document_interne",
            })
        return formater_documents([d for d, _ in resultats])

    @tool
    def recherche_web(requete: str) -> str:
        """Recherche des informations recentes sur le web, priorite aux sites officiels UVCI."""
        resultats = rechercher_sur_le_web(requete, nb_resultats=4, cibler_uvci=True)
        for r in resultats:
            sources_collectees.append({
                "fichier": r.url,
                "nom_court": r.url,
                "page": "web",
                "score": 80.0,
                "type": "web",
                "titre": r.titre,
            })
        return formater_resultats_web(resultats)

    outils = [recherche_documents_internes, recherche_web]
    outils_par_nom = {o.name: o for o in outils}
    modele = creer_modele_chat().bind_tools(outils)

    messages = [SystemMessage(content=_get_prompt())]
    messages.extend(historique)
    messages.append(HumanMessage(content=question))

    reponse_finale = "je n'ai pas pu traiter votre question, reessayez"

    for _ in range(MAX_TOURS_AGENT):
        try:
            resultat = modele.invoke(messages)
        except Exception as e:
            logger.error(f"erreur appel llm: {e}")
            break

        messages.append(resultat)

        if not resultat.tool_calls:
            reponse_finale = resultat.content
            break

        for appel in resultat.tool_calls:
            outil = outils_par_nom.get(appel["name"])
            if outil is None:
                sortie = "outil inconnu"
            else:
                try:
                    sortie = outil.invoke(appel["args"])
                except Exception as e:
                    logger.error(f"erreur execution outil {appel['name']}: {e}")
                    sortie = f"erreur outil: {e}"
            messages.append(ToolMessage(content=str(sortie), tool_call_id=appel["id"]))
    else:
        # boucle epuisee sans reponse finale, on force une derniere generation sans outils
        try:
            reponse_finale = creer_modele_chat().invoke(messages).content
        except Exception as e:
            logger.error(f"erreur generation finale: {e}")

    sources_dedupliquees = _dedupliquer_sources(sources_collectees)

    return {
        "reponse": reponse_finale,
        "nb_sources": len(sources_dedupliquees),
        "sources": sources_dedupliquees,
    }


def _dedupliquer_sources(sources: list) -> list:
    vues = set()
    resultat = []
    for s in sources:
        cle = (s["fichier"], s.get("page"))
        if cle in vues:
            continue
        vues.add(cle)
        resultat.append(s)
    return resultat


def _prompt_par_defaut() -> str:
    return """Tu es Eduheures, l'assistant virtuel de l'Universite Virtuelle de Cote d'Ivoire (UVCI).
Nous sommes en 2026.

Tu disposes de deux outils: recherche_documents_internes pour la base documentaire UVCI,
et recherche_web pour les informations recentes ou absentes de la base interne.

Regles:
Appelle recherche_documents_internes pour toute question sur formations, inscriptions, frais, calendriers, contacts.
Appelle recherche_web pour les actualites, evenements recents, ou si la base interne ne suffit pas.
Tu peux appeler les deux outils, dans n'importe quel ordre, et plusieurs fois si necessaire.
Utilise uniquement les informations retournees par les outils. N'invente jamais un chiffre ou un nom.
Si aucune information pertinente n'est trouvee, dis-le et oriente vers le site officiel de uvci et leurs adresse email scolarite@uvci.edu.ci.
Reponds en francais, ton professionnel et bienveillant, reponse concise sauf si un detail exhaustif est demande.
Cite tes sources en fin de reponse: nom de fichier et page pour un document interne, titre et url pour une source web."""