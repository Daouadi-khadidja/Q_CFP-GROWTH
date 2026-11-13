# CFP-Growth : trouve les itemsets fréquents avec un seuil (MIS) différent par produit.
from collections import defaultdict

from mis import compter_supports, lire_mis_txt


# Arrête le programme quand il y a trop d'itemsets.
class TropDeMotifs(Exception):
    pass


# Lit la taille maximale : un nombre, ou "n" / None pour aucune limite.
def lire_taille_max(valeur):
    if valeur is None or str(valeur).strip().lower() in ("", "n"):
        return None
    return int(valeur)


# Représente un nœud de l'arbre.
class Noeud:
    def __init__(self, item, parent):
        self.item = item
        self.compte = 0
        self.parent = parent
        self.enfants = {}


# Ajoute un chemin dans l'arbre.
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


# Renvoie les produits situés au-dessus d'un nœud.
def chemin_vers_racine(noeud):
    chemin = []
    parent = noeud.parent
    while parent.item is not None:
        chemin.append(parent.item)
        parent = parent.parent
    chemin.reverse()
    return chemin


# Fouille l'arbre récursivement.
def fouiller_arbre(entete, rang, suffixe, seuil_min, resultats, stats):
    for item in sorted(entete, key=lambda p: -rang[p]):
        noeuds = entete[item]
        groupe = suffixe + (item,)
        resultats[frozenset(groupe)] = sum(n.compte for n in noeuds)
        if stats["limite"] and len(resultats) > stats["limite"]:
            raise TropDeMotifs(len(resultats))
        if stats["taille_max"] and len(groupe) >= stats["taille_max"]:
            continue

        base = []
        compteurs = defaultdict(int)
        for n in noeuds:
            chemin = chemin_vers_racine(n)
            if chemin:
                base.append((chemin, n.compte))
                for p in chemin:
                    compteurs[p] += n.compte
        stats["bases"] += 1

        gardes = {p for p, c in compteurs.items() if c >= seuil_min}
        if not gardes:
            continue
        petite_racine = Noeud(None, None)
        petit_entete = defaultdict(list)
        for chemin, compte in base:
            chemin_filtre = [p for p in chemin if p in gardes]
            if chemin_filtre:
                inserer_chemin(petite_racine, chemin_filtre, compte, petit_entete, stats)
        fouiller_arbre(petit_entete, rang, groupe, seuil_min, resultats, stats)


# Lance CFP-Growth avec les seuils (en tickets) de chaque produit.
def cfp_growth_seuils(tickets, seuils, mis_tri=None, taille_max=None, limite_motifs=None):
    tickets_ens = [set(t) for t in tickets]
    if mis_tri is None:
        mis_tri = seuils

    nb = compter_supports(tickets_ens)
    MIN = min(seuils[p] for p in nb)
    gardes = [p for p in nb if nb[p] >= MIN]
    ordre = sorted(gardes, key=lambda p: (-mis_tri[p], -nb[p], p))
    rang = {produit: position for position, produit in enumerate(ordre)}

    racine = Noeud(None, None)
    entete = defaultdict(list)
    stats = {"noeuds": 0, "bases": 0, "limite": limite_motifs, "taille_max": taille_max}
    for ticket in tickets_ens:
        chemin = sorted((p for p in ticket if p in rang), key=lambda p: rang[p])
        if chemin:
            inserer_chemin(racine, chemin, 1, entete, stats)

    brut = {}
    fouiller_arbre(entete, rang, (), MIN, brut, stats)

    frequents = {}
    for groupe, support in brut.items():
        if support >= min(seuils[p] for p in groupe):
            frequents[tuple(sorted(groupe))] = support

    return {"frequents": frequents, "MIN": MIN, "ordre_arbre": ordre}


# Lit mis.txt puis lance CFP-Growth.
def cfp_growth(tickets, fichier_mis, taille_max=None, limite_motifs=None):
    mis, seuils = lire_mis_txt(fichier_mis)
    manquants = {p for t in tickets for p in t} - set(seuils)
    if manquants:
        raise ValueError("mis.txt ne correspond pas aux données : %s" % sorted(manquants)[:5])
    res = cfp_growth_seuils(tickets, seuils, mis_tri=mis,
                            taille_max=lire_taille_max(taille_max), limite_motifs=limite_motifs)
    res["mis"] = mis
    res["seuils"] = seuils
    return res