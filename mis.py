# mis.py : calcule le MIS (seuil minimum) de chaque produit et l'ÉCRIT dans un fichier mis.txt.
# Ensuite cfp_growth.py LIT ce fichier : les deux programmes ne se parlent que par mis.txt.
#
# Vocabulaire :
#   ticket  = une ligne du dataset (la liste de ses items)
#   support = nombre de tickets qui contiennent un produit
#   MIS     = seuil minimum propre à chaque produit (en nombre de tickets)
#   LS      = plancher du MIS, en fraction des tickets (0.10 = 10 %)
import math
from collections import defaultdict

SEP = "\t"                                    # séparateur des colonnes dans mis.txt (une tabulation)
NOMS_CATEGORIES = ("faible", "moyenne", "forte")


# 1. Compte, pour chaque produit, le nombre de tickets qui le contiennent.
def compter_supports(tickets):
    nb = defaultdict(int)
    for ticket in tickets:
        for produit in set(ticket):
            nb[produit] += 1
    return dict(nb)


# 2. Quantile (Q1, Q3...) par interpolation linéaire.
def quantile_interpolation(valeurs, p):
    """p = 0.25 donne Q1, p = 0.75 donne Q3. Position = 1 + (n-1) x p, puis on interpole entre deux valeurs voisines."""
    v = sorted(valeurs)
    n = len(v)
    pos = 1 + (n - 1) * p
    i = int(math.floor(pos))
    reste = pos - i
    if i >= n:
        return float(v[-1])
    return v[i - 1] + reste * (v[i] - v[i - 1])


# 3. Catégorie de chaque produit : 0 = faible, 1 = moyenne, 2 = forte.
def categories_quartiles(nb, niveaux=(0.25, 0.75)):
    """Renvoie (coupures Q1 et Q3, catégorie de chaque produit)."""
    supports = list(nb.values())
    coupures = [quantile_interpolation(supports, p) for p in niveaux]
    categories = {}
    for produit in nb:
        categories[produit] = sum(1 for coupure in coupures if nb[produit] >= coupure)
    return coupures, categories


# 4. Beta automatique : un beta par catégorie.
def calculer_betas(nb, ls, niveaux=(0.25, 0.75)):
    """beta = LS + (1 - LS) x (s - s_min) / (s_max - s_min), avec
    s = Q1 (faible), (Q1 + Q3) / 2 (moyenne), Q3 (forte)."""
    supports = list(nb.values())
    smin, smax = min(supports), max(supports)
    q1, q3 = [quantile_interpolation(supports, p) for p in niveaux]
    etendue = smax - smin
    betas = []
    for s in (q1, (q1 + q3) / 2, q3):
        x = 0.0 if etendue == 0 else (s - smin) / etendue      # évite la division par 0
        betas.append(ls + (1 - ls) * x)
    return tuple(betas)


# 5. LS automatique (plancher du MIS), en fraction des tickets.
def calculer_ls_auto(nb, N, niveau=0.75):
    """LS = max(1 / racine(N), Q3 des supports / N)."""
    q3 = quantile_interpolation(list(nb.values()), niveau)
    return max(1 / math.sqrt(N), q3 / N)


# 6. Calcul complet du MIS de chaque produit.
def calculer_mis(tickets, ls=None, niveaux=(0.25, 0.75)):
    N = len(tickets)
    nb = compter_supports(tickets)

    if ls is None:
        ls = calculer_ls_auto(nb, N)

    betas = calculer_betas(nb, ls, niveaux)
    coupures, categories = categories_quartiles(nb, niveaux)

    mis = {}
    seuils = {}
    supports_rel = {}

    for produit in nb:
        support_rel = nb[produit] / N
        beta = betas[categories[produit]]

        mis_rel = max(ls, beta * support_rel)

        supports_rel[produit] = support_rel
        mis[produit] = mis_rel

        # Conversion uniquement pour l'algorithme
        seuils[produit] = math.ceil(mis_rel * N)

    return {
        "N": N,
        "support": nb,
        "support_rel": supports_rel,
        "ls": ls,
        "mis": mis,
        "seuils": seuils,
        "categories": categories,
        "coupures": coupures,
        "betas": betas
    }

# 7. ÉCRIT mis.txt : une ligne par produit.
def ecrire_mis_txt(m, chemin):
    N = m["N"]

    with open(chemin, "w", encoding="utf-8") as f:
        f.write("# MIS en proportions relatives\n")
        f.write("# N = %d tickets\n" % N)
        f.write("# LS = %.6f (%.2f%%)\n" %
                (m["ls"], 100 * m["ls"]))

        f.write("# Q1 = %.6f, Q3 = %.6f\n" %
                (m["coupures"][0] / N,
                 m["coupures"][1] / N))

        for nom, beta in zip(NOMS_CATEGORIES, m["betas"]):
            f.write("# beta %s = %.4f\n" % (nom, beta))

        f.write("# produit\tsupport_rel\tcategorie\tbeta\tMIS_rel\tseuil_tickets\n")

        for produit in sorted(
            m["mis"],
            key=lambda p: (-m["mis"][p], p)
        ):
            categorie = m["categories"][produit]

            ligne = [
                produit,
                "%.9f" % m["support_rel"][produit],
                NOMS_CATEGORIES[categorie],
                "%.4f" % m["betas"][categorie],
                "%.9f" % m["mis"][produit],
                str(m["seuils"][produit])
            ]

            f.write(SEP.join(ligne) + "\n")


# 8. LIT mis.txt (utilisé par cfp_growth.py).
def lire_mis_txt(chemin):
    """Renvoie (mis, seuils) : deux dictionnaires {produit: valeur}."""
    mis = {}
    seuils = {}
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.rstrip("\n")
            if ligne.strip() == "" or ligne.startswith("#"):
                continue                               # on saute les commentaires et les lignes vides
            parties = ligne.split(SEP)
            produit = parties[0]
            mis[produit] = float(parties[4])
            seuils[produit] = int(parties[5])
    return mis, seuils