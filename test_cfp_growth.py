# =====================================================================
#  TESTS : CFP-Growth trouve-t-il exactement les bons itemsets ?
#   1. exemple du PDF (Hu & Chen) : comparaison avec un calcul direct (force brute)
#   2. petits tickets du cours : CFP-Growth contre MIS-Apriori
#   3. groceries2.csv (sous-ensembles) : CFP-Growth contre MIS-Apriori
#  Lancer :   python test_cfp_growth.py
# =====================================================================
import itertools
import math
import os

from cfp_growth import cfp_growth, cfp_growth_seuils
from mis_apriori import mis_apriori


def force_brute(tickets, seuils):
    """Tous les groupes X avec Sup(X) >= plus petit MIS de ses produits (définition directe)."""
    ens = [set(t) for t in tickets]
    produits = sorted(set().union(*ens))
    sortie = {}
    for k in range(1, len(produits) + 1):
        for g in itertools.combinations(produits, k):
            sup = sum(1 for t in ens if set(g) <= t)
            if sup >= min(seuils[p] for p in g):
                sortie[g] = sup
    return sortie


def verifier(nom, a, b):
    ok = a == b
    print(("OK    " if ok else "ERREUR"), nom, "-", len(a), "itemsets", "" if ok else "(%d contre %d)" % (len(a), len(b)))
    if not ok:
        print("   seulement dans le premier :", sorted(set(a) - set(b))[:5])
        print("   seulement dans le second  :", sorted(set(b) - set(a))[:5])
    return ok


tout_ok = True

# 1. exemple du PDF
tickets = [list("dcaf"), list("gcafe"), list("bacfh"), list("gbf"), list("bc")]
seuils = dict(a=4, b=4, c=4, d=3, e=3, f=2, g=2, h=2)
cfp = cfp_growth_seuils(tickets, seuils)["frequents"]
tout_ok &= verifier("Exemple du PDF : CFP-Growth contre force brute", cfp, force_brute(tickets, seuils))
print("   abf fréquent ?", ("a", "b", "f") in cfp, "(le PDF le donne fréquent, mais son support est 1 < 2)")

# 2. petits tickets du cours
t5 = [["Pain", "Lait", "Thé"], ["Pain", "Lait"], ["Pain", "Oeufs"], ["Lait", "Thé"], ["Pain", "Lait", "Thé"]]
tout_ok &= verifier("5 tickets du cours : CFP-Growth contre MIS-Apriori",
                    cfp_growth(t5, 0.2, 0.5)["frequents"], mis_apriori(t5, 0.2, 0.5)["frequents"])

# 3. groceries
fichier = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groceries2.csv")
if os.path.exists(fichier):
    tickets = []
    for ligne in open(fichier, encoding="utf-8"):
        p = sorted({x.strip() for x in ligne.split(",") if x.strip()})
        if p:
            tickets.append(p)
    for n in (500, 2000):
        for ls, beta in ((0.0025, 0.2), (0.005, 0.5)):
            tout_ok &= verifier("groceries (%d tickets, LS=%g, beta=%g)" % (n, ls, beta),
                                cfp_growth(tickets[:n], ls, beta)["frequents"],
                                mis_apriori(tickets[:n], ls, beta)["frequents"])
print("\nRésultat global :", "TOUS LES TESTS PASSENT" if tout_ok else "IL Y A DES ERREURS")
