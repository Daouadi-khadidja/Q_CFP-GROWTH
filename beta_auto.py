# =====================================================================
#  CHOIX AUTOMATIQUE DE BETA  (amélioration proposée : idée A)
#
#  Problème : beta (dans MIS(i) = max(LS, beta x Sup(i))) est choisi "à la main".
#  Idée : on essaie plusieurs beta avec CFP-growth (rapide) et on choisit
#  le PLUS PETIT beta qui respecte deux conditions :
#     1. BUDGET  : nombre d'itemsets <= budget   (sinon trop de résultats)
#     2. QUALITÉ : proportion de paires à lift > 1 >= cible
#  Le plus petit beta = celui qui garde le plus de produits rares.
#
#  Limites (à dire au prof) :
#   - le lift > 1 monte presque tout seul quand beta monte : d'où le budget
#   - la qualité est mesurée sur les PAIRES seulement
#   - la cible et le budget sont encore des choix humains (on a déplacé le
#     réglage de beta vers un critère plus lisible, on ne l'a pas supprimé)
#
#  Utilisation :
#      beta, tableau = choisir_beta(tickets, 0.0025, betas=[0.1, 0.2, 0.5],
#                                   budget=2000, cible=0.95)
#      (LS, la liste des beta, le budget et la cible sont donnés par l'utilisateur)
# =====================================================================
from cfp_growth import cfp_growth


def qualite_paires(res, N):
    """Proportion de paires fréquentes dont le lift est > 1 (None s'il n'y a pas de paire)."""
    nb = res["support"]
    lifts = []
    for g, s in res["frequents"].items():
        if len(g) == 2:
            lifts.append(s * N / (nb[g[0]] * nb[g[1]]))
    if not lifts:
        return None, 0
    return sum(1 for l in lifts if l > 1) / len(lifts), len(lifts)


def choisir_beta(tickets, ls, betas, budget=2000, cible=0.95):
    """betas : liste de valeurs de beta À TESTER, fournie par l'utilisateur.
    Le programme n'invente aucune valeur : il choisit parmi celles du client."""
    betas = sorted(betas)
    N = len(tickets)
    tableau = []
    for b in betas:
        res = cfp_growth(tickets, ls, b)
        q, nb_paires = qualite_paires(res, N)
        tableau.append({"beta": b, "itemsets": len(res["frequents"]),
                        "paires": nb_paires, "lift_sup_1": q})
    # conditions : budget respecté ET qualité atteinte -> le plus petit beta
    for ligne in tableau:
        if ligne["itemsets"] <= budget and ligne["lift_sup_1"] is not None \
                and ligne["lift_sup_1"] >= cible:
            return ligne["beta"], tableau
    # sinon : parmi ceux qui respectent le budget, la meilleure qualité
    ok = [l for l in tableau if l["itemsets"] <= budget and l["lift_sup_1"] is not None]
    if ok:
        return max(ok, key=lambda l: l["lift_sup_1"])["beta"], tableau
    return betas[-1], tableau


if __name__ == "__main__":
    # exemple minuscule (5 tickets du cours)
    t5 = [["Pain", "Lait", "Thé"], ["Pain", "Lait"], ["Pain", "Oeufs"],
          ["Lait", "Thé"], ["Pain", "Lait", "Thé"]]
    b, tab = choisir_beta(t5, 0.2, betas=[0.2, 0.5, 0.8], budget=50, cible=0.5)
    for l in tab:
        print(l)
    print("beta choisi :", b)
