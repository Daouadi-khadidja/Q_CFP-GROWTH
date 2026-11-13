# Calcule le MIS de chaque produit (formule catégories + ancres, sans LS) et lit/écrit mis.txt.
import math
from collections import defaultdict

SEP = "\t"
NOMS_CATEGORIES = ("faible", "moyenne", "forte")


# Compte le nombre de tickets qui contiennent chaque produit.
def compter_supports(tickets):
    nb = defaultdict(int)
    for ticket in tickets:
        for produit in set(ticket):
            nb[produit] += 1
    return dict(nb)


# Calcule un quantile par interpolation linéaire.
def quantile_interpolation(valeurs, p):
    v = sorted(valeurs)
    n = len(v)
    pos = 1 + (n - 1) * p
    i = int(math.floor(pos))
    if i >= n:
        return float(v[-1])
    return v[i - 1] + (pos - i) * (v[i] - v[i - 1])


# Donne la catégorie de chaque produit : 0 faible, 1 moyenne, 2 forte.
def categories_quartiles(nb, niveaux=(0.25, 0.75)):
    supports = list(nb.values())
    coupures = [quantile_interpolation(supports, p) for p in niveaux]
    categories = {p: sum(1 for c in coupures if nb[p] >= c) for p in nb}
    return coupures, categories


# Calcule les bêtas, les MIS et les seuils de chaque produit.
def calculer_mis_categories(tickets, niveaux=(0.25, 0.75)):
    N = len(tickets)
    nb = compter_supports(tickets)
    supports = list(nb.values())
    smin, smax = min(supports), max(supports)
    q1, q3 = [quantile_interpolation(supports, p) for p in niveaux]
    coupures, categories = categories_quartiles(nb, niveaux)

    beta_min, beta_max = smin / smax, q3 / smax
    etendue = smax - smin
    betas = []
    for r in (q1, (q1 + q3) / 2, q3):
        x = 0.0 if etendue == 0 else (r - smin) / etendue
        betas.append(beta_min + (beta_max - beta_min) * x)
    betas = tuple(betas)

    mis, seuils, supports_rel, betas_item = {}, {}, {}, {}
    for produit in nb:
        supports_rel[produit] = nb[produit] / N
        betas_item[produit] = betas[categories[produit]]
        mis[produit] = betas_item[produit] * supports_rel[produit]
        seuils[produit] = math.ceil(round(mis[produit] * N, 9))

    return {"N": N, "support": nb, "support_rel": supports_rel, "mis": mis, "seuils": seuils,
            "categories": categories, "coupures": coupures, "betas_item": betas_item, "betas": betas}


# Écrit mis.txt : une ligne par produit.
def ecrire_mis_txt(m, chemin):
    N = m["N"]
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("# MIS en proportions relatives\n")
        f.write("# N = %d tickets\n" % N)
        f.write("# Q1 = %.6f, Q3 = %.6f\n" % (m["coupures"][0] / N, m["coupures"][1] / N))
        for nom, beta in zip(("faible", "moyen", "fort"), m["betas"]):
            f.write("# beta %s = %.5f\n" % (nom, beta))
        f.write("# produit\tsupport_rel\tcategorie\tbeta\tMIS_rel\tseuil_tickets\n")
        for produit in sorted(m["mis"], key=lambda p: (-m["mis"][p], p)):
            f.write(SEP.join([produit,
                              "%.9f" % m["support_rel"][produit],
                              NOMS_CATEGORIES[m["categories"][produit]],
                              "%.5f" % m["betas_item"][produit],
                              "%.9f" % m["mis"][produit],
                              str(m["seuils"][produit])]) + "\n")


# Lit mis.txt et renvoie (mis, seuils).
def lire_mis_txt(chemin):
    mis, seuils = {}, {}
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.rstrip("\n")
            if ligne.strip() == "" or ligne.startswith("#"):
                continue
            parties = ligne.split(SEP)
            mis[parties[0]] = float(parties[4])
            seuils[parties[0]] = int(parties[5])
    return mis, seuils