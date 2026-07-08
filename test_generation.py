from src.generation import repondre
import json

if __name__ == "__main__":
    question = "donne tout les filiere dispoiinible pour la licence ?"
    resultat = repondre(question, [])
    print(json.dumps(resultat, ensure_ascii=False, indent=2))
