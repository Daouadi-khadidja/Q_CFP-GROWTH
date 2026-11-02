# CFP-Growth (MIS par quartiles) : itemsets fréquents avec un seuil différent par produit (MIS = max(LS x N, beta x Sup)).
import math
from collections import defaultdict

import numpy as np


# 1. Noeud : une case de l'arbre (produit, compteur, parent, enfants).
class Noeud:
    def __init__(self, item, parent):
        self.item = item
        self.compte = 0
        self.parent = parent
        self.enfants = {}


# 2. Ajoute un chemin dans l'arbre en réutilisant les nœuds communs et en augmentant leur compteur.
def inserer_chemin(racine, chemin, compte, entete, stats):
    noeud = racine
    for item in chemin:
        if item in noeud.enfants:
            enfant = noeud.enfants[item]
        else:
            enfant = Noeud(item, noeud)
            noeud.enfants[item] = enfant
            entete[item].append(enfant)
            stats["noeuds"] += 1
        enfant.compte += compte
        noeud = enfant


def chemin_vers_racine(noeud):
    """Liste des produits situés au-dessus d'un nœud, de la racine à son parent."""
    chemin = []
    parent = noeud.parent
    while parent.item is not None:
        chemin.append(parent.item)
        parent = parent.parent
    chemin.reverse()
    return chemin


# 3. Fouille récursive : pour chaque produit (du bas vers le haut), note son support puis explore sa base conditionnelle.
def fouiller_arbre(entete, rang, suffixe, seuil_min, resultats, stats):
    produits = sorted(entete, key=lambda p: -rang[p])

    for item in produits:
        noeuds = entete[item]

        # a) Support du groupe (suffixe + item).
        support = 0
        for n in noeuds:
            support += n.compte
        groupe = suffixe + (item,)
        resultats[frozenset(groupe)] = support

        # b) Base conditionnelle : les chemins qui montent vers la racine.
        base = []
        compteurs = defaultdict(int)
        for n in noeuds:
            chemin = chemin_vers_racine(n)
            if chemin:
                base.append((chemin, n.compte))
                for p in chemin:
                    compteurs[p] += n.compte
        stats["bases"] += 1

        # c) Garder les produits assez fréquents et construire le petit arbre conditionnel.
        gardes = {p for p, c in compteurs.items() if c >= seuil_min}
        if not gardes:
            continue

        petite_racine = Noeud(None, None)
        petit_entete = defaultdict(list)
        for chemin, compte in base:
            chemin_filtre = [p for p in chemin if p in gardes]
            if chemin_filtre:
                inserer_chemin(petite_racine, chemin_filtre, compte, petit_entete, stats)

        # d) Recommencer dans le petit arbre.
        fouiller_arbre(petit_entete, rang, groupe, seuil_min, resultats, stats)


# CFP-Growth quand le seuil (MIS en nombre de tickets) de chaque produit est déjà connu.
def cfp_growth_seuils(tickets, seuils, mis_tri=None):
    tickets_ens = [set(t) for t in tickets]
    if mis_tri is None:
        mis_tri = seuils

    # Étape 1 : support de chaque produit (premier passage).
    nb = compter_supports(tickets_ens)

    # Étape 2 : MIN = plus petit seuil, un produit sous MIN ne sera jamais fréquent.
    MIN = min(seuils[p] for p in nb)
    gardes = [p for p in nb if nb[p] >= MIN]

    # Étape 3 : ordre des produits par MIS décroissant (puis support décroissant, puis alphabétique).
    ordre = sorted(gardes, key=lambda p: (-mis_tri[p], -nb[p], p))
    rang = {produit: position for position, produit in enumerate(ordre)}

    # Étape 4 : construire l'arbre (deuxième passage).
    racine = Noeud(None, None)
    entete = defaultdict(list)
    stats = {"noeuds": 0, "bases": 0}
    occurrences = 0
    for ticket in tickets_ens:
        chemin = [p for p in ticket if p in rang]
        chemin.sort(key=lambda p: rang[p])
        occurrences += len(chemin)
        if chemin:
            inserer_chemin(racine, chemin, 1, entete, stats)
    noeuds_arbre = stats["noeuds"]

    # Étape 5 : fouiller l'arbre pour trouver tous les groupes qui atteignent MIN.
    brut = {}
    fouiller_arbre(entete, rang, (), MIN, brut, stats)

    # Étape 6 : test final, un groupe est fréquent si son support >= le plus petit MIS de ses produits.
    frequents = {}
    par_niveau = defaultdict(list)
    enumeres_par_niveau = defaultdict(int)
    for groupe, support in brut.items():
        taille = len(groupe)
        enumeres_par_niveau[taille] += 1
        plus_petit_mis = min(seuils[p] for p in groupe)
        if support >= plus_petit_mis:
            noms = tuple(sorted(groupe))
            frequents[noms] = support
            par_niveau[taille].append(noms)
    for taille in par_niveau:
        par_niveau[taille].sort()

    return {"frequents": frequents,
            "frequents_par_niveau": dict(par_niveau),
            "enumeres_par_niveau": dict(enumeres_par_niveau),
            "passages": 2,
            "noeuds_arbre": noeuds_arbre,
            "occurrences": occurrences,
            "bases_conditionnelles": stats["bases"],
            "noeuds_total": stats["noeuds"],
            "MIN": MIN, "ordre_arbre": ordre}


# 4. Renvoie {produit: nombre de tickets qui le contiennent}.
def compter_supports(tickets):
    nb = defaultdict(int)
    for ticket in tickets:
        for produit in set(ticket):
            nb[produit] += 1
    return dict(nb)


# 5. MIS par quartiles : catégorie faible / moyenne / forte selon Q1 et Q2, un beta par catégorie.
def categories_quartiles(nb, niveaux=(0.25, 0.75)):
    """Renvoie (coupures, catégorie de chaque produit) avec 0 = faible, 1 = moyenne, 2 = forte."""
    supports = list(nb.values())
    coupures = [float(np.percentile(supports, 100 * p)) for p in niveaux]
    categories = {}
    for produit in nb:
        categories[produit] = sum(1 for coupure in coupures if nb[produit] >= coupure)
    return coupures, categories


def calculer_mis_quartiles(nb, N, ls, betas=(0.2, 0.5, 0.8), niveaux=(0.25, 0.75)):
    """MIS de chaque produit en nombre de tickets : max(LS x N, beta de sa catégorie x support)."""
    if len(betas) != len(niveaux) + 1:
        raise ValueError("il faut un beta de plus que de niveaux (3 beta pour Q1 et Q2)")
    if any(betas[i] > betas[i + 1] for i in range(len(betas) - 1)):
        raise ValueError("les beta doivent être en ordre croissant")
    if not all(0 < b <= 1 for b in betas):
        raise ValueError("chaque beta doit être dans ]0, 1]")

    coupures, categories = categories_quartiles(nb, niveaux)
    mis = {}
    for produit in nb:
        beta = betas[categories[produit]]
        mis[produit] = max(round(ls * N, 9), round(beta * nb[produit], 9))
    return mis, categories, coupures


def cfp_growth(tickets, ls, betas=(0.2, 0.5, 0.8), niveaux=(0.25, 0.75)):
    """Fonction principale : CFP-Growth avec MIS par quartiles (ls = plancher en fraction, 3 beta croissants)."""
    N = len(tickets)
    nb = compter_supports(tickets)
    mis, categories, coupures = calculer_mis_quartiles(nb, N, ls, betas, niveaux)
    seuils = {p: math.ceil(mis[p]) for p in nb}

    res = cfp_growth_seuils(tickets, seuils, mis_tri=mis)
    res["mis"] = {p: mis[p] / N for p in nb}
    res["support"] = nb
    res["seuils"] = seuils
    res["categories"] = categories
    res["coupures"] = coupures
    res["betas"] = tuple(betas)
    return res