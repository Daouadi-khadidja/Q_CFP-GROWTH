# =====================================================================
#  APRIORI ORIGINAL (comme dans le cours, sans aucune amélioration)
#
#  - un seul seuil MinSup pour tous les produits
#  - recherche par niveau : produits seuls, puis paires, puis trios...
#  - le support de chaque candidat se calcule en relisant tous les tickets
#
#  Utilisation :
#      resultat = apriori_original(tickets, 0.4)
#      tickets : liste de tickets, un ticket = liste de produits
#      minsup  : seuil en fraction (0.4 = 40 % des tickets)
# =====================================================================
import math


def apriori_original(tickets, minsup):
    N = len(tickets)
    seuil = math.ceil(round(minsup * N, 9))        # nombre minimum de tickets
    tickets_ens = [set(t) for t in tickets]

    # Étape 1 : les candidats de taille 1 sont tous les produits
    produits = sorted(set().union(*tickets_ens))
    candidats = [(p,) for p in produits]

    frequents = {}                 # {groupe : nombre de tickets} tous niveaux confondus
    candidats_par_niveau = {}      # {k : liste des candidats testés au niveau k}
    frequents_par_niveau = {}      # {k : liste des groupes fréquents du niveau k}
    k = 1

    # Tant qu'il reste des candidats
    while len(candidats) > 0:
        candidats_par_niveau[k] = list(candidats)

        # Étape 2 : calculer le support de chaque candidat (on relit les tickets)
        # Étape 3 : garder seulement ceux dont le support est >= MinSup
        frequents_k = []
        for groupe in candidats:
            ens = set(groupe)
            compteur = 0
            for ticket in tickets_ens:
                if ens <= ticket:                  # le ticket contient tout le groupe
                    compteur += 1
            if compteur >= seuil:
                frequents[groupe] = compteur
                frequents_k.append(groupe)
        frequents_par_niveau[k] = list(frequents_k)

        # Étape 4 : construire les candidats du niveau suivant
        #   a) jointure : on colle deux groupes qui ont le même début
        #   b) élagage  : tous les sous-groupes du candidat doivent être fréquents
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


# ---------------------------------------------------------------------
#  Exemple : les 5 tickets du cours MIS (Pain, Lait, Thé, Oeufs)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    tickets = [["Pain", "Lait", "Thé"], ["Pain", "Lait"], ["Pain", "Oeufs"],
               ["Lait", "Thé"], ["Pain", "Lait", "Thé"]]
    res = apriori_original(tickets, 0.4)          # MinSup = 40 %
    for k in res["candidats"]:
        print("Niveau", k, "- candidats :", res["candidats"][k])
        print("          fréquents :", res["frequents_par_niveau"][k])
