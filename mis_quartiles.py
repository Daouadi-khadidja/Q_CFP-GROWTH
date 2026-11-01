# =====================================================================
#  MIS AUTOMATIQUE PAR QUARTILES  (idée de la camarade + formule de Liu)
#
#  1. on calcule le support de chaque produit
#  2. on calcule les quartiles Q1, Q2 (méthode du rang le plus proche)
#  3. on classe chaque produit :   faible    : support < Q1
#                                  moyenne   : Q1 <= support < Q2
#                                  forte     : support >= Q2
#  4. chaque catégorie a son propre beta (beta_faible <= beta_moyen <= beta_fort <= 1) :
#          MIS(i) = max( LS , beta_catégorie(i) x Sup(i) )
#  5. on lance CFP-Growth avec ces seuils
#
#  Avec des niveaux (0.25, 0.5, 0.75) et 4 beta, on obtient 4 catégories (Q1, Q2, Q3).
#
#  Utilisation :
#      res = cfp_growth_quartiles(tickets, ls=0.015, betas=(0.2, 0.5, 0.8))
#      res = cfp_growth_quartiles(tickets, 0.015, (0.1, 0.3, 0.6, 0.9), niveaux=(0.25, 0.5, 0.75))
# =====================================================================
import math
from collections import defaultdict

from cfp_growth import cfp_growth_seuils


def quantile_rang_plus_proche(valeurs_triees, p):
    """Méthode du rang le plus proche : valeur à la position ceil(p x n) (positions de 1 à n)."""
    n = len(valeurs_triees)
    position = max(1, math.ceil(round(p * n, 9)))
    return valeurs_triees[position - 1]


def categories_quartiles(nb, niveaux=(0.25, 0.5)):
    """nb[produit] = nombre de tickets. Renvoie (coupures, catégorie de chaque produit).
    catégorie = nombre de coupures <= support (0 = faible, 1 = moyenne, 2 = forte...)."""
    valeurs = sorted(nb.values())
    coupures = [quantile_rang_plus_proche(valeurs, p) for p in niveaux]
    categories = {p: sum(1 for q in coupures if nb[p] >= q) for p in nb}
    return coupures, categories


def mis_par_quartiles(nb, N, ls, betas=(0.2, 0.5, 0.8), niveaux=(0.25, 0.5)):
    """MIS(i) = max(LS, beta_catégorie x Sup(i)), en nombre de tickets."""
    if len(betas) != len(niveaux) + 1:
        raise ValueError("il faut un beta de plus que de niveaux (3 beta pour Q1 et Q2)")
    if any(betas[i] > betas[i + 1] for i in range(len(betas) - 1)):
        raise ValueError("les beta doivent être en ordre croissant")
    if not all(0 < b <= 1 for b in betas):
        raise ValueError("chaque beta doit être dans ]0, 1]")
    coupures, categories = categories_quartiles(nb, niveaux)
    mis = {p: max(round(ls * N, 9), round(betas[categories[p]] * nb[p], 9)) for p in nb}
    return mis, categories, coupures


def cfp_growth_quartiles(tickets, ls, betas=(0.2, 0.5, 0.8), niveaux=(0.25, 0.5)):
    N = len(tickets)
    nb = defaultdict(int)
    for t in tickets:
        for p in set(t):
            nb[p] += 1
    nb = dict(nb)
    mis, categories, coupures = mis_par_quartiles(nb, N, ls, betas, niveaux)
    seuils = {p: math.ceil(mis[p]) for p in nb}
    res = cfp_growth_seuils(tickets, seuils, mis_tri=mis)
    res.update({"mis": {p: mis[p] / N for p in nb}, "support": nb, "seuils": seuils,
                "categories": categories, "coupures": coupures, "betas": tuple(betas)})
    return res


if __name__ == "__main__":
    # Exemple des captures : supports 90, 80, 60, 20, 10 % (N = 10 tickets), LS = 5 %
    nb = {"A": 9, "B": 8, "C": 6, "D": 2, "E": 1}
    mis, cat, q = mis_par_quartiles(nb, 10, 0.05)
    print("Q1 = %d %%, Q2 = %d %%" % (10 * q[0], 10 * q[1]))
    for p in "EDCBA":
        print(p, "support", 10 * nb[p], "%", "| catégorie", ("faible", "moyenne", "forte")[cat[p]],
              "| MIS = %.0f %%" % (10 * mis[p]))
