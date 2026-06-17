import os
import re
import logging
import dotenv
dotenv.load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from src.config import FICHIER_PROMPT
from src.utils import creer_modele_chat, lire_prompt, formater_documents
from src.retrieval import extraire_signal_recherche, recuperer_documents, charger_base_qdrant
from src.web_search import (
    rechercher_sur_le_web,
    formater_resultats_web,
    extraire_signal_recherche_web,
)

logger = logging.getLogger(__name__)

#  Caches singletons 
_prompt_cache = None
_base_cache   = None

# Mots-clés qui forcent la recherche web même sans signal explicite
MOTS_CLES_ACTUALITE = (
    "actualité", "actualités", "récent", "récente", "nouveau", "nouvelle",
    "2025", "2026", "dernière", "dernières", "news", "événement",
    "annonce", "annonces", "agenda", "calendrier 2026",
)


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


#  Construction des messages

def construire_messages(prompt_systeme: str, historique: list, contenu_utilisateur: str) -> list:
    messages = [SystemMessage(content=prompt_systeme)]
    messages.extend(historique)
    messages.append(HumanMessage(content=contenu_utilisateur))
    return messages


# Passes de génération 

def premiere_passe(question: str, historique: list) -> str:
    """Le LLM décide s'il faut chercher dans Qdrant et/ou sur le web."""
    modele = creer_modele_chat()
    prompt_systeme = _get_prompt()
    messages = construire_messages(prompt_systeme, historique, question)
    resultat = modele.invoke(messages)
    return resultat.content


def deuxieme_passe(
    question: str,
    documents_qdrant: list,
    resultats_web: list,
    historique: list,
) -> str:
    """
    Répond en enrichissant avec les documents Qdrant ET les résultats web.
    Demande explicitement au LLM de citer UNIQUEMENT les sources utilisées.
    """
    modele = creer_modele_chat()
    prompt_systeme = _get_prompt()

    parties_contexte = []

    if documents_qdrant:
        contexte_qdrant = formater_documents(documents_qdrant)
        parties_contexte.append(
            f"## Documents internes UVCI\n\n{contexte_qdrant}"
        )

    if resultats_web:
        contexte_web = formater_resultats_web(resultats_web)
        parties_contexte.append(
            f"## Résultats web récents\n\n{contexte_web}"
        )

    contexte_complet = "\n\n".join(parties_contexte) if parties_contexte else "Aucun document trouvé."

    contenu_utilisateur = (
        f"Question : {question}\n\n"
        f"Contexte :\n{contexte_complet}\n\n"
        f"INSTRUCTION CRITIQUE : Réponds uniquement à partir du contexte ci-dessus. "
        f"Si le contexte ne contient pas l'information, dis-le explicitement et oriente "
        f"vers scolarite@uvci.edu.ci. "
        f"Dans ta réponse, cite en ligne les sources que tu as réellement utilisées "
        f"(nom du fichier et/ou URL). Ne cite aucune source que tu n'as pas lue dans ce contexte."
    )

    messages = construire_messages(prompt_systeme, historique, contenu_utilisateur)
    resultat = modele.invoke(messages)
    return resultat.content


# Filtrage des sources citées 

def _filtrer_sources_citees(sources: list, reponse: str, cle: str = "nom_court") -> list:
   
    citees = []
    reponse_lower = reponse.lower()
    for source in sources:
        nom = source.get(cle, "")
        nom_sans_ext = os.path.splitext(nom)[0].lower()
        if nom.lower() in reponse_lower or (nom_sans_ext and nom_sans_ext in reponse_lower):
            citees.append(source)
    return citees


# Orchestrateur principal 

def repondre(question: str, historique: list = None) -> dict:
    
    if historique is None:
        historique = []

    question_porte_sur_actualite = any(
        mot in question.lower() for mot in MOTS_CLES_ACTUALITE
    )

    # Chargement Qdrant avec fallback gracieux
    try:
        base = _get_base()
        base_disponible = True
    except Exception as e:
        logger.warning(f"Base Qdrant non disponible : {e}")
        base = None
        base_disponible = False

    # Étape 1 : première passe 
    reponse_brute = premiere_passe(question, historique)

    query_qdrant, reponse_temp   = extraire_signal_recherche(reponse_brute)
    query_web,    reponse_finale = extraire_signal_recherche_web(reponse_temp)

    sources_qdrant   = []
    resultats_web    = []
    documents_qdrant = []

    # Étape 2 : recherche Qdrant
    if query_qdrant and base_disponible:
        try:
            resultats_qdrant = recuperer_documents(query_qdrant, base)
            documents_qdrant = [doc for doc, _ in resultats_qdrant]

            for doc, score in resultats_qdrant:
                try:
                    score_confiance = round((1 - score) * 100, 1) if isinstance(score, (int, float)) else 0
                except Exception:
                    score_confiance = 0
                score_confiance = max(0, min(100, score_confiance))

                sources_qdrant.append({
                    "fichier":   doc.metadata.get("source", "Inconnu"),
                    "nom_court": os.path.basename(doc.metadata.get("source", "Inconnu")),
                    "page":      doc.metadata.get("page", "N/A"),
                    "score":     score_confiance,
                    "type":      "document_interne",
                })

            if sources_qdrant:
                confiance_moyenne = sum(s["score"] for s in sources_qdrant) / len(sources_qdrant)
                if confiance_moyenne < 50:
                    logger.info(f"Confiance Qdrant faible ({confiance_moyenne:.1f}%) — résultats écartés")
                    sources_qdrant = []
                    documents_qdrant = []

        except Exception as e:
            logger.error(f"Erreur recherche Qdrant : {e}")

    #  Étape 3 : recherche web 
    doit_chercher_web = (
        bool(query_web)
        or (query_qdrant and not documents_qdrant)
        or question_porte_sur_actualite
    )

    if doit_chercher_web:
        query_web_effective = query_web or query_qdrant or question
        if question_porte_sur_actualite and "2026" not in query_web_effective:
            query_web_effective = f"{query_web_effective} 2026"
        try:
            resultats_web = rechercher_sur_le_web(
                query=query_web_effective,
                nb_resultats=4,
                cibler_uvci=True,
            )
            logger.info(f"Recherche web : {len(resultats_web)} résultat(s) pour '{query_web_effective[:60]}'")
        except Exception as e:
            logger.error(f"Erreur recherche web : {e}")
            resultats_web = []

    #  Étape 4 : deuxième passe 
    if documents_qdrant or resultats_web:
        try:
            reponse_finale = deuxieme_passe(question, documents_qdrant, resultats_web, historique)
        except Exception as e:
            logger.error(f"Erreur deuxième passe : {e}")

    # Nettoyage des signaux résiduels
    reponse_finale = re.sub(r"\[RECHERCHE[^\]]*\]", "", reponse_finale, flags=re.IGNORECASE).strip()

    # Étape 5 : filtrage des sources réellement citées ─────────────────────
    sources_qdrant_filtrees = _filtrer_sources_citees(sources_qdrant, reponse_finale, cle="nom_court")

    sources_web_filtrees = [
        {
            "fichier":   r.url,
            "nom_court": r.url,
            "page":      "web",
            "score":     80.0,
            "type":      "web",
            "titre":     r.titre,
        }
        for r in resultats_web
        if r.url in reponse_finale or r.titre in reponse_finale
    ]

    # Fallback : aucune citation détectée → meilleur doc Qdrant uniquement
    if not sources_qdrant_filtrees and sources_qdrant:
        meilleure = max(sources_qdrant, key=lambda s: s["score"])
        sources_qdrant_filtrees = [meilleure]
        logger.info("Fallback : source la plus pertinente retournée")

    toutes_sources = sources_qdrant_filtrees + sources_web_filtrees

    return {
        "reponse":    reponse_finale,
        "nb_sources": len(toutes_sources),
        "sources":    toutes_sources,
    }


# Prompt de secours 

def _prompt_par_defaut() -> str:
    return """Tu es Eduheures, l'assistant virtuel de l'Université Virtuelle de Côte d'Ivoire (UVCI).

Tu aides les étudiants et futurs étudiants sur : formations, inscriptions, frais, calendriers, contacts.

RÈGLES :
1. Utilise uniquement les informations du contexte fourni.
2. N'invente aucune information.
3. Si tu n'as pas l'information, oriente vers : scolarite@uvci.edu.ci ou https://uvci.online
4. Réponds toujours en français, avec un ton professionnel et bienveillant.
5. Cite uniquement les sources que tu as réellement utilisées."""