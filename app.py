from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import os

from src.generation import repondre
from src.config import MAX_HISTORIQUE

load_dotenv()

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        EDUHEURES — Assistant Universitaire UVCI          ║
║    Université Virtuelle de Côte d'Ivoire                 ║
╚══════════════════════════════════════════════════════════╝
"""


def lancer_assistant():
    print(BANNER)
    print("Initialisation de l'assistant...")

    historique = []

    print("\nBonjour ! Je suis Eduheures, votre assistant universitaire de l'UVCI.")
    print("Je peux vous renseigner sur les formations, inscriptions, calendriers, et bien plus.")
    print("Tapez 'quitter' pour terminer.\n")

    while True:
        question = input("Vous : ").strip()

        if not question:
            continue

        if question.lower() in ("quitter", "exit", "quit"):
            print("\nMerci d'avoir utilisé Eduheures. Bonne continuation dans vos études !")
            break

        resultat       = repondre(question, historique)
        reponse_finale = resultat["reponse"]
        sources        = resultat["sources"]

        print(f"\nEduheures : {reponse_finale}")

        if sources:
            print("\n Sources utilisées :")
            for i, source in enumerate(sources, 1):
                type_src = source.get("type", "document")
                if type_src == "web":
                    titre = source.get("titre") or source.get("fichier", "Source web")
                    url   = source.get("fichier", "")
                    print(f"  [{i}] {titre}")
                    if url:
                        print(f"       {url}")
                else:
                    nom   = os.path.basename(source.get("fichier", "Inconnu"))
                    score = source.get("score", 0)
                    page  = source.get("page", "")
                    page_str = f", p.{int(page)+1}" if isinstance(page, (int, float)) else ""
                    print(f"  [{i}] {nom}{page_str} (confiance : {score}%)")

        print()

        historique.append(HumanMessage(content=question))
        historique.append(AIMessage(content=reponse_finale))

        if len(historique) > MAX_HISTORIQUE * 2:
            historique = historique[-(MAX_HISTORIQUE * 2):]


if __name__ == "__main__":
    lancer_assistant()