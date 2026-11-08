# CFP-Growth : trouve les itemsets fréquents avec un seuil (MIS) différent par produit.
# Les MIS ne sont PAS calculés ici : ils sont lus dans le fichier mis.txt écrit par mis.py.
from collections import defaultdict

from mis import compter_supports, lire_mis_txt


# 0. Garde-fou facultatif : levée quand le nombre de groupes énumérés dépasse `limite_motifs` (explosion mémoire).
class TropDeMotifs(Exception):
    pass


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
        if stats.get("limite") and len(resultats) > stats["limite"]:
            raise TropDeMotifs(len(resultats))

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


# 4. CFP-Growth quand le seuil (en nombre de tickets) de chaque produit est déjà connu.
def cfp_growth_seuils(tickets, seuils, mis_tri=None, limite_motifs=None):
    """seuils = {produit: nombre minimum de tickets}. mis_tri = valeurs utilisées pour ordonner les produits."""
    tickets_ens = [set(t) for t in tickets]
    if mis_tri is None:
        mis_tri = seuils

    # Étape 1 : support de chaque produit (premier passage sur les tickets).
    nb = compter_supports(tickets_ens)

    # Étape 2 : MIN = plus petit seuil. Un produit sous MIN ne sera jamais fréquent.
    MIN = min(seuils[p] for p in nb)
    gardes = [p for p in nb if nb[p] >= MIN]

    # Étape 3 : ordre des produits par MIS décroissant (puis support décroissant, puis alphabétique).
    ordre = sorted(gardes, key=lambda p: (-mis_tri[p], -nb[p], p))
    rang = {produit: position for position, produit in enumerate(ordre)}

    # Étape 4 : construire l'arbre (deuxième passage).
    racine = Noeud(None, None)
    entete = defaultdict(list)
    stats = {"noeuds": 0, "bases": 0, "limite": limite_motifs}
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

    # Étape 6 : test final, un groupe est fréquent si son support >= le plus petit seuil de ses produits.
    frequents = {}
    par_niveau = defaultdict(list)
    enumeres_par_niveau = defaultdict(int)
    for groupe, support in brut.items():
        taille = len(groupe)
        enumeres_par_niveau[taille] += 1
        plus_petit_seuil = min(seuils[p] for p in groupe)
        if support >= plus_petit_seuil:
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


# 5. Fonction principale : lit mis.txt (écrit par mis.py) puis lance CFP-Growth.
def cfp_growth(tickets, fichier_mis, limite_motifs=None):
    mis, seuils = lire_mis_txt(fichier_mis)

    # Vérification : chaque produit des tickets doit avoir une ligne dans mis.txt.
    produits = {p for t in tickets for p in t}
    manquants = produits - set(seuils)
    if manquants:
        raise ValueError("mis.txt ne correspond pas aux données (produits absents : %s). "
                         "Relance main.py pour régénérer mis.txt." % sorted(manquants)[:5])

    res = cfp_growth_seuils(tickets, seuils, mis_tri=mis, limite_motifs=limite_motifs)
    res["mis"] = mis
    res["seuils"] = seuils
    return res