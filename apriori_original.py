# Apriori original (cours) : un seul MinSup, recherche par niveau, support recalculé en relisant les tickets.
import math


def apriori_original(tickets, minsup):
    """Apriori classique ; minsup en fraction (0.4 = 40 % des tickets)."""
    N = len(tickets)
    seuil = math.ceil(round(minsup * N, 9))
    tickets_ens = [set(t) for t in tickets]

    # Étape 1 : les candidats de taille 1 sont tous les produits.
    produits = sorted(set().union(*tickets_ens))
    candidats = [(p,) for p in produits]

    frequents = {}
    candidats_par_niveau = {}
    frequents_par_niveau = {}
    k = 1

    while len(candidats) > 0:
        candidats_par_niveau[k] = list(candidats)

        # Étapes 2 et 3 : compter le support de chaque candidat et garder ceux >= MinSup.
        frequents_k = []
        for groupe in candidats:
            ens = set(groupe)
            compteur = 0
            for ticket in tickets_ens:
                if ens <= ticket:
                    compteur += 1
            if compteur >= seuil:
                frequents[groupe] = compteur
                frequents_k.append(groupe)
        frequents_par_niveau[k] = list(frequents_k)

        # Étape 4 : candidats du niveau suivant par jointure puis élagage des sous-groupes non fréquents.
        frequents_k.sort()
        fréquents_ensemble = set(frequents_k)
        debuts = {}
        for groupe in frequents_k:
            debuts.setdefault(groupe[:-1], []).append(groupe[-1])

        candidats = []
        for debut, fins in debuts.items():
            for x in range(len(fins)):
                for y in range(x + 1, len(fins)):
                    candidat = debut + (fins[x], fins[y])
                    ok = True
                    for i in range(len(candidat)):
                        sous_groupe = candidat[:i] + candidat[i + 1:]
                        if sous_groupe not in fréquents_ensemble:
                            ok = False
                            break
                    if ok:
                        candidats.append(candidat)
        candidats.sort()
        k += 1

    return {"frequents": frequents,
            "candidats": candidats_par_niveau,
            "frequents_par_niveau": frequents_par_niveau}


# Exemple : les 5 tickets du cours MIS avec MinSup = 40 %.
if __name__ == "__main__":
    tickets = [["Pain", "Lait", "Thé"], ["Pain", "Lait"], ["Pain", "Oeufs"],
               ["Lait", "Thé"], ["Pain", "Lait", "Thé"]]
    res = apriori_original(tickets, 0.4)
    for k in res["candidats"]:
        print("Niveau", k, "- candidats :", res["candidats"][k])
        print("          fréquents :", res["frequents_par_niveau"][k])