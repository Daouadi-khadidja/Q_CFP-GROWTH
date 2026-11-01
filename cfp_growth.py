# =====================================================================
#  CFP_GROWTH.PY  -  l'algorithme CFP-Growth (MIS calculé par quartiles)
#
#  But : trouver les groupes de produits (itemsets) achetés souvent ensemble,
#        avec un seuil DIFFÉRENT pour chaque produit (le MIS).
#
#  Sommaire du fichier :
#     1. Node               : la structure de l'arbre
#     2. Insert a path      : ajouter un chemin dans l'arbre
#     3. Mine the tree      : fouiller l'arbre pour trouver les itemsets
#     4. Item supports      : compter combien de tickets contiennent chaque produit
#     5. MIS with quartiles : calculer le MIS de chaque produit, puis lancer CFP-Growth
#
#  Vocabulaire :
#     ticket  = liste de produits achetés ensemble (une ligne du dataset)
#     support = nombre de tickets qui contiennent le produit (ou le groupe)
#     MIS     = support minimum demandé pour UN produit
#     LS      = plancher : aucun MIS ne descend en dessous de LS
#     beta    = part du support du produit utilisée pour son MIS
#               MIS(i) = max( LS x N , beta_catégorie(i) x Sup(i) )
#
#  Ce fichier ne se lance pas seul : il est utilisé par main.py et comparison.py.
# =====================================================================
import math
from collections import defaultdict

import numpy as np


# =====================================================================
#  1. NODE : la structure de l'arbre
#     Les tickets sont rangés dans un arbre : les débuts communs sont partagés,
#     ce qui compresse la base.
# =====================================================================
class Noeud:
    """Une case de l'arbre : un produit, combien de tickets passent par là, son parent et ses enfants."""

    def __init__(self, item, parent):
        self.item = item          # le produit (None pour la racine)
        self.compte = 0           # nombre de tickets qui passent par ce nœud
        self.parent = parent      # le nœud au-dessus
        self.enfants = {}         # {produit: nœud enfant}


# =====================================================================
#  2. INSERT A PATH : ajouter un chemin dans l'arbre
# =====================================================================
def inserer_chemin(racine, chemin, compte, entete, stats):
    """Ajoute un chemin (liste de produits) dans l'arbre.
    Si le début du chemin existe déjà, on réutilise les nœuds et on augmente leur compteur."""
    noeud = racine
    for item in chemin:
        if item in noeud.enfants:
            enfant = noeud.enfants[item]               # le nœud existe déjà
        else:
            enfant = Noeud(item, noeud)                # on crée un nouveau nœud
            noeud.enfants[item] = enfant
            entete[item].append(enfant)                # on le note dans la table d'en-tête
            stats["noeuds"] += 1
        enfant.compte += compte
        noeud = enfant


def chemin_vers_racine(noeud):
    """Liste des produits situés au-dessus d'un nœud (de la racine jusqu'à son parent)."""
    chemin = []
    parent = noeud.parent
    while parent.item is not None:                     # on monte jusqu'à la racine
        chemin.append(parent.item)
        parent = parent.parent
    chemin.reverse()                                   # ordre : racine -> bas
    return chemin


# =====================================================================
#  3. MINE THE TREE : fouiller l'arbre
# =====================================================================
def fouiller_arbre(entete, rang, suffixe, seuil_min, resultats, stats):
    """Fonction qui s'appelle elle-même. Pour chaque produit de la table d'en-tête
    (en partant du bas de l'arbre) :
         a) on note son support,
         b) on regarde les chemins qui mènent à lui (la "base conditionnelle"),
         c) on construit un petit arbre avec ces chemins,
         d) on recommence dans ce petit arbre pour trouver les groupes plus grands.
       suffixe = produits déjà choisis (le groupe en cours de construction)."""
    produits = sorted(entete, key=lambda p: -rang[p])  # du bas de l'arbre vers le haut

    for item in produits:
        noeuds = entete[item]

        # a) support du groupe (suffixe + item)
        support = 0
        for n in noeuds:
            support += n.compte
        groupe = suffixe + (item,)
        resultats[frozenset(groupe)] = support

        # b) base conditionnelle : les chemins qui montent vers la racine
        base = []                                      # liste de (chemin, compte)
        compteurs = defaultdict(int)                   # combien de fois chaque produit apparaît
        for n in noeuds:
            chemin = chemin_vers_racine(n)
            if chemin:
                base.append((chemin, n.compte))
                for p in chemin:
                    compteurs[p] += n.compte
        stats["bases"] += 1

        # c) on garde seulement les produits assez fréquents dans cette base
        gardes = {p for p, c in compteurs.items() if c >= seuil_min}
        if not gardes:
            continue                                   # rien à explorer plus loin

        petite_racine = Noeud(None, None)
        petit_entete = defaultdict(list)
        for chemin, compte in base:
            chemin_filtre = [p for p in chemin if p in gardes]
            if chemin_filtre:
                inserer_chemin(petite_racine, chemin_filtre, compte, petit_entete, stats)

        # d) on recommence dans le petit arbre
        fouiller_arbre(petit_entete, rang, groupe, seuil_min, resultats, stats)


def cfp_growth_seuils(tickets, seuils, mis_tri=None):
    """CFP-Growth quand on connaît déjà le seuil de chaque produit.
    tickets : liste de tickets (chaque ticket = liste de produits)
    seuils  : {produit: MIS en nombre de tickets (entier)}
    mis_tri : valeurs utilisées pour ranger les produits (par défaut : seuils)"""
    tickets_ens = [set(t) for t in tickets]            # set = pas de doublon dans un ticket
    if mis_tri is None:
        mis_tri = seuils

    # Étape 1 : support de chaque produit (premier passage sur les données)
    nb = compter_supports(tickets_ens)

    # Étape 2 : MIN = le plus petit seuil. Un produit sous MIN ne pourra jamais être fréquent.
    MIN = min(seuils[p] for p in nb)
    gardes = [p for p in nb if nb[p] >= MIN]

    # Étape 3 : ordre des produits = MIS décroissant (à égalité : ordre alphabétique)
    ordre = sorted(gardes, key=lambda p: (-mis_tri[p], p))
    rang = {produit: position for position, produit in enumerate(ordre)}

    # Étape 4 : construire l'arbre (deuxième passage sur les données)
    racine = Noeud(None, None)
    entete = defaultdict(list)
    stats = {"noeuds": 0, "bases": 0}
    occurrences = 0
    for ticket in tickets_ens:
        chemin = [p for p in ticket if p in rang]      # on enlève les produits supprimés
        chemin.sort(key=lambda p: rang[p])             # on les range dans l'ordre de l'arbre
        occurrences += len(chemin)
        if chemin:
            inserer_chemin(racine, chemin, 1, entete, stats)
    noeuds_arbre = stats["noeuds"]

    # Étape 5 : fouiller l'arbre -> tous les groupes qui atteignent au moins MIN
    brut = {}
    fouiller_arbre(entete, rang, (), MIN, brut, stats)

    # Étape 6 : test final. Un groupe est fréquent si son support >= le plus petit MIS de ses produits
    frequents = {}                                     # {("a", "b"): support}
    par_niveau = defaultdict(list)                     # groupes fréquents, classés par taille
    enumeres_par_niveau = defaultdict(int)             # groupes trouvés avant le test final
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
            "passages": 2,                             # 1 passage pour les supports, 1 pour l'arbre
            "noeuds_arbre": noeuds_arbre,
            "occurrences": occurrences,                # produits gardés dans tous les tickets
            "bases_conditionnelles": stats["bases"],
            "noeuds_total": stats["noeuds"],           # arbre + petits arbres
            "MIN": MIN, "ordre_arbre": ordre}


# =====================================================================
#  4. ITEM SUPPORTS : compter les produits
# =====================================================================
def compter_supports(tickets):
    """Renvoie {produit: nombre de tickets qui le contiennent}."""
    nb = defaultdict(int)
    for ticket in tickets:
        for produit in set(ticket):
            nb[produit] += 1
    return dict(nb)


# =====================================================================
#  5. MIS WITH QUARTILES
#
#     1. on calcule le support de chaque produit
#     2. on calcule Q1 et Q2 avec np.percentile
#     3. catégorie du produit :  faible  : support < Q1
#                                moyenne : Q1 <= support < Q2
#                                forte   : support >= Q2
#     4. un beta par catégorie (beta_faible <= beta_moyen <= beta_fort <= 1) :
#             MIS(i) = max( LS x N , beta_catégorie(i) x Sup(i) )
#
#  Avec niveaux = (0.25, 0.5, 0.75) et 4 beta, on obtient 4 catégories.
# =====================================================================
def categories_quartiles(nb, niveaux=(0.25, 0.75)):
    """Renvoie (coupures, catégorie de chaque produit).
    catégorie = nombre de coupures que le support atteint (0 = faible, 1 = moyenne, 2 = forte...)."""
    supports = list(nb.values())
    # np.percentile attend un pourcentage de 0 à 100 : 0.25 devient 25
    coupures = [float(np.percentile(supports, 100 * p)) for p in niveaux]
    categories = {}
    for produit in nb:
        categories[produit] = sum(1 for coupure in coupures if nb[produit] >= coupure)
    return coupures, categories


def calculer_mis_quartiles(nb, N, ls, betas=(0.2, 0.5, 0.8), niveaux=(0.25, 0.75)):
    """MIS de chaque produit selon sa catégorie, en nombre de tickets."""
    # vérifications des paramètres
    if len(betas) != len(niveaux) + 1:
        raise ValueError("il faut un beta de plus que de niveaux (3 beta pour Q1 et Q2)")
    if any(betas[i] > betas[i + 1] for i in range(len(betas) - 1)):
        raise ValueError("les beta doivent être en ordre croissant")
    if not all(0 < b <= 1 for b in betas):
        raise ValueError("chaque beta doit être dans ]0, 1]")

    coupures, categories = categories_quartiles(nb, niveaux)
    mis = {}
    for produit in nb:
        beta = betas[categories[produit]]              # le beta de sa catégorie
        # round(..., 9) évite les petites erreurs de calcul des décimaux
        mis[produit] = max(round(ls * N, 9), round(beta * nb[produit], 9))
    return mis, categories, coupures


def cfp_growth(tickets, ls, betas=(0.2, 0.5, 0.8), niveaux=(0.25, 0.75)):
    """LA fonction à appeler : CFP-Growth avec MIS par quartiles.
    tickets : liste de tickets
    ls      : plancher du support (fraction, ex : 0.1 = 10 %)
    betas   : 3 beta croissants (produits faibles, moyens, forts)"""
    N = len(tickets)
    nb = compter_supports(tickets)
    mis, categories, coupures = calculer_mis_quartiles(nb, N, ls, betas, niveaux)
    seuils = {p: math.ceil(mis[p]) for p in nb}        # nombre entier de tickets (arrondi vers le haut)

    res = cfp_growth_seuils(tickets, seuils, mis_tri=mis)
    res["mis"] = {p: mis[p] / N for p in nb}           # MIS en fraction (0 à 1)
    res["support"] = nb
    res["seuils"] = seuils
    res["categories"] = categories
    res["coupures"] = coupures
    res["betas"] = tuple(betas)
    return res