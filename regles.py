# Règles d'association SI ... ALORS ... à partir des itemsets fréquents.
# support = Sup(G+C)/N, confiance = Sup(G+C)/Sup(G), lift = confiance / (Sup(C)/N).
from itertools import combinations


def preparer_bitsets(tickets):
    """Chaque item devient un grand entier binaire (1 bit par ticket) pour compter vite."""
    bits = {}
    for i, ticket in enumerate(tickets):
        for item in set(ticket):
            bits[item] = bits.get(item, 0) | (1 << i)
    return bits


def calculer_support(items, bitsets, deja_calcule):
    """Nombre de tickets contenant tous les items (résultats mis en cache)."""
    cle = tuple(sorted(items))
    if cle not in deja_calcule:
        nombre = bitsets[cle[0]]
        for item in cle[1:]:
            nombre &= bitsets[item]
        deja_calcule[cle] = nombre.bit_count()
    return deja_calcule[cle]



def generer_regles(frequents, N, bitsets, confiance_min=0.6, lift_min=1.0,
                   conclusion_max=1, conclusion_contient="", separateur="  ET  "):
    """
    Génère les règles d'association avec des supports et confiances
    exprimés en proportions (entre 0 et 1).
    """

    deja_calcule = dict(frequents)
    mot = conclusion_contient.lower()
    regles = []

    for groupe, sup in frequents.items():
        taille = len(groupe)

        if taille < 2:
            continue

        limite = taille - 1 if conclusion_max == 0 else min(
            conclusion_max, taille - 1
        )

        for k in range(1, limite + 1):
            for conclusion in combinations(groupe, k):

                if mot and not all(mot in c.lower() for c in conclusion):
                    continue

                gauche = [i for i in groupe if i not in conclusion]

                # Supports relatifs
                support_groupe = sup / N

                support_gauche = (
                    calculer_support(gauche, bitsets, deja_calcule) / N
                )

                support_conclusion = (
                    calculer_support(conclusion, bitsets, deja_calcule) / N
                )

                # Confiance et lift
                confiance = support_groupe / support_gauche

                if confiance < confiance_min:
                    continue

                lift = confiance / support_conclusion

                if lift <= lift_min + 1e-12:
                    continue

                regles.append({
                    "si": separateur.join(gauche),
                    "alors": separateur.join(conclusion),

                    # Proportions entre 0 et 1
                    "support": round(support_groupe, 6),
                    "confiance": round(confiance, 6),

                    # Mesure sans unité
                    "lift": round(lift, 6)
                })

    regles.sort(
        key=lambda r: (-r["lift"], -r["confiance"])
    )

    return regles
