
import math
from collections import defaultdict


class Noeud:
    __slots__ = ("item", "compte", "parent", "enfants")

    def __init__(self, item, parent):
        self.item = item
        self.compte = 0
        self.parent = parent
        self.enfants = {}


def _inserer(racine, chemin, compte, entete, stats):
    """Insère un chemin dans l'arbre : les préfixes communs sont partagés."""
    noeud = racine
    for item in chemin:
        enfant = noeud.enfants.get(item)
        if enfant is None:
            enfant = Noeud(item, noeud)
            noeud.enfants[item] = enfant
            entete[item].append(enfant)          # node-link de la table d'en-tête
            stats["noeuds"] += 1
        enfant.compte += compte
        noeud = enfant


def _miner(entete, rang, suffixe, seuil_min, sortie, stats):
    """Parcourt la table d'en-tête depuis le bas et construit les bases conditionnelles."""
    for item in sorted(entete, key=lambda i: -rang[i]):
        noeuds = entete[item]
        sup = sum(n.compte for n in noeuds)
        motif = suffixe + (item,)
        sortie[frozenset(motif)] = sup           # motif MIN-fréquent
        stats["enumeres"] += 1

        # base conditionnelle : chemins qui montent vers la racine, sans l'item
        base = []
        cnt = defaultdict(int)
        for n in noeuds:
            chemin = []
            p = n.parent
            while p.item is not None:
                chemin.append(p.item)
                p = p.parent
            if chemin:
                chemin.reverse()
                base.append((chemin, n.compte))
                for it in chemin:
                    cnt[it] += n.compte
        stats["bases"] += 1

        # arbre conditionnel : on garde les items dont le support conditionnel >= MIN
        gardes = {it for it, c in cnt.items() if c >= seuil_min}
        if not gardes:
            continue
        racine = Noeud(None, None)
        ent = defaultdict(list)
        for chemin, c in base:
            ch = [it for it in chemin if it in gardes]
            if ch:
                _inserer(racine, ch, c, ent, stats)
        _miner(ent, rang, motif, seuil_min, sortie, stats)


def cfp_growth_seuils(tickets, seuils, mis_tri=None):
    """CFP-Growth avec des seuils MIS donnés en nombre de tickets.
    seuils[p] = MIS de p (entier). mis_tri[p] = valeur servant au tri (par défaut : seuils)."""
    tickets_ens = [set(t) for t in tickets]
    mis_tri = mis_tri or seuils

    # 1. supports (premier passage sur la base)
    nb = defaultdict(int)
    for t in tickets_ens:
        for p in t:
            nb[p] += 1
    MIN = min(seuils[p] for p in nb)

    # 2. supprimer les produits sous MIN
    gardes = [p for p in nb if nb[p] >= MIN]
    # 3. ordre : MIS décroissant (à égalité : ordre alphabétique)
    ordre = sorted(gardes, key=lambda p: (-mis_tri[p], p))
    rang = {p: r for r, p in enumerate(ordre)}

    # 4. MIS-tree (deuxième passage sur la base)
    racine = Noeud(None, None)
    entete = defaultdict(list)
    stats = {"noeuds": 0, "enumeres": 0, "bases": 0}
    occurrences = 0
    for t in tickets_ens:
        chemin = sorted((p for p in t if p in rang), key=lambda p: rang[p])
        occurrences += len(chemin)
        if chemin:
            _inserer(racine, chemin, 1, entete, stats)
    noeuds_arbre = stats["noeuds"]

    # 5-6. bases conditionnelles et arbres conditionnels (motifs MIN-fréquents)
    brut = {}
    _miner(entete, rang, (), MIN, brut, stats)

    # 7. test final : Sup(X) >= MIS(X) = plus petit MIS des produits de X
    frequents = {}
    par_niveau = defaultdict(list)
    enumeres_par_niveau = defaultdict(int)
    for motif, sup in brut.items():
        k = len(motif)
        enumeres_par_niveau[k] += 1
        if sup >= min(seuils[p] for p in motif):
            noms = tuple(sorted(motif))
            frequents[noms] = sup
            par_niveau[k].append(noms)
    for k in par_niveau:
        par_niveau[k].sort()

    return {"frequents": frequents,
            "frequents_par_niveau": dict(par_niveau),
            "enumeres_par_niveau": dict(enumeres_par_niveau),   # motifs MIN-fréquents (avant le test final)
            "passages": 2,                                      # 1 pour les supports, 1 pour l'arbre
            "noeuds_arbre": noeuds_arbre,
            "occurrences": occurrences,                         # produits gardés dans tous les tickets
            "bases_conditionnelles": stats["bases"],
            "noeuds_total": stats["noeuds"],                    # arbre + arbres conditionnels
            "MIN": MIN, "ordre_arbre": ordre}


def cfp_growth(tickets, ls, beta):
    """Même formule MIS que mis_apriori.py : MIS(i) = max(LS, beta x Sup(i))."""
    N = len(tickets)
    nb = defaultdict(int)
    for t in tickets:
        for p in set(t):
            nb[p] += 1
    mis = {p: max(round(ls * N, 9), round(beta * nb[p], 9)) for p in nb}
    seuils = {p: math.ceil(mis[p]) for p in nb}
    res = cfp_growth_seuils(tickets, seuils, mis_tri=mis)
    res["mis"] = {p: mis[p] / N for p in nb}
    res["support"] = dict(nb)
    return res


if __name__ == "__main__":
    # Exemple de Hu & Chen (2006) : supports en nombre de transactions
    tickets = [list("dcaf"), list("gcafe"), list("bacfh"), list("gbf"), list("bc")]
    seuils = dict(a=4, b=4, c=4, d=3, e=3, f=2, g=2, h=2)
    res = cfp_growth_seuils(tickets, seuils)
    print("MIN =", res["MIN"], "| ordre de l'arbre :", res["ordre_arbre"])
    print("Nœuds du MIS-tree :", res["noeuds_arbre"])
    for k in sorted(res["frequents_par_niveau"]):
        print("Niveau", k, ":", [("".join(g), res["frequents"][g]) for g in res["frequents_par_niveau"][k]])
