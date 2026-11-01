# =====================================================================
#  TESTS du MIS par quartiles
#   1. exemple des captures : Q1, Q2, catégories et MIS
#   2. CFP-Growth avec ces seuils contre un calcul direct (force brute)
#   3. contrôles d'erreur sur les beta
#  Lancer :   python test_mis_quartiles.py
# =====================================================================
import itertools
import os

from mis_quartiles import cfp_growth_quartiles, mis_par_quartiles, quantile_rang_plus_proche

ok_global = True


def verifier(nom, condition):
    global ok_global
    ok_global &= bool(condition)
    print("OK    " if condition else "ERREUR", nom)


# 1. exemple des captures (supports 90, 80, 60, 20, 10 % ; N = 10)
nb = {"A": 9, "B": 8, "C": 6, "D": 2, "E": 1}
mis, cat, q = mis_par_quartiles(nb, 10, 0.05)
verifier("Q1 = 20 % et Q2 = 60 %", q == [2, 6])
verifier("catégories : E faible, D moyenne, C B A forte", [cat[p] for p in "EDCBA"] == [0, 1, 2, 2, 2])
verifier("MIS = 5, 10, 48, 64, 72 %", [round(100 * mis[p] / 10) for p in "EDCBA"] == [5, 10, 48, 64, 72])
verifier("rang le plus proche : 5 valeurs, p = 0,25 -> position 2", quantile_rang_plus_proche([1, 2, 6, 8, 9], 0.25) == 2)


# 2. CFP-Growth avec les seuils par quartiles contre la force brute
def force_brute(tickets, seuils):
    ens = [set(t) for t in tickets]
    produits = sorted(set().union(*ens))
    sortie = {}
    for k in range(1, len(produits) + 1):
        for g in itertools.combinations(produits, k):
            sup = sum(1 for t in ens if set(g) <= t)
            if sup >= min(seuils[p] for p in g):
                sortie[g] = sup
    return sortie


t5 = [["Pain", "Lait", "Thé"], ["Pain", "Lait"], ["Pain", "Oeufs"], ["Lait", "Thé"], ["Pain", "Lait", "Thé"]]
r = cfp_growth_quartiles(t5, 0.2)
verifier("5 tickets : CFP-Growth = force brute", r["frequents"] == force_brute(t5, r["seuils"]))

fichier = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groceries2.csv")
if os.path.exists(fichier):
    tous = []
    for l in open(fichier, encoding="utf-8"):
        p = sorted({x.strip() for x in l.split(",") if x.strip()})
        if p:
            tous.append(p)
    freq = {}
    for t in tous:
        for p in t:
            freq[p] = freq.get(p, 0) + 1
    top = set(sorted(freq, key=lambda p: -freq[p])[:12])          # 12 produits : la force brute reste faisable
    petits = [[p for p in t if p in top] for t in tous[:2000]]
    petits = [t for t in petits if t]
    for betas, niv in (((0.2, 0.5, 0.8), (0.25, 0.5)), ((0.1, 0.3, 0.6, 0.9), (0.25, 0.5, 0.75))):
        r = cfp_growth_quartiles(petits, 0.01, betas, niv)
        verifier("groceries (12 produits, %d catégories) : CFP-Growth = force brute" % (len(niv) + 1),
                 r["frequents"] == force_brute(petits, r["seuils"]))

# 3. contrôles d'erreur
for betas, nom in (((0.8, 0.5, 0.2), "beta décroissants refusés"), ((0.2, 0.5), "mauvais nombre de beta refusé"),
                   ((0.2, 0.5, 1.5), "beta > 1 refusé")):
    try:
        mis_par_quartiles(nb, 10, 0.05, betas)
        verifier(nom, False)
    except ValueError:
        verifier(nom, True)

print("\nRésultat global :", "TOUS LES TESTS PASSENT" if ok_global else "IL Y A DES ERREURS")
