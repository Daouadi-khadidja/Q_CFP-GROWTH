# Compare Apriori (calculé dans WEKA, résultats importés) avec CFP-Growth (python comparison.py).
#
# Marche à suivre :
#   1. Dans Weka (Explorer > Associate > Apriori), mettre : outputItemSets = True,
#      lowerBoundMinSupport = upperBoundMinSupport = LS (affiché par ce programme), numRules = 100000.
#   2. Copier le texte de la sortie Weka dans le fichier FICHIER_WEKA (voir réglages ci-dessous).
#   3. Lancer python comparison.py : il vérifie que CFP-Growth trouve les mêmes itemsets que Weka,
#      puis compare Weka (un seul seuil) avec CFP-Growth (un MIS par produit, lu dans mis.txt).
import math
import os
import statistics
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cfp_growth import cfp_growth, cfp_growth_seuils
from main import lire_tickets, nettoyer
from mis import calculer_ls_auto, calculer_mis, compter_supports, ecrire_mis_txt

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# RÉGLAGES : les seules valeurs à modifier avant de relancer.
FICHIER = os.path.join(DOSSIER, "Data", "Lim_solar_data_net.csv")
FICHIER_WEKA = os.path.join(DOSSIER, "Weka", "apriori_weka.txt")        # sortie texte copiée depuis Weka
DOSSIER_SORTIE = os.path.join(DOSSIER, "resultats_comparaison")
FICHIER_MIS = os.path.join(DOSSIER_SORTIE, "mis.txt")

LS = None                    # None = LS automatique. Mets la MÊME valeur que le minsup donné à Weka (0.10 = 10 %)
TEMPS_WEKA = None            # temps de Weka en secondes si tu l'as mesuré (sinon None)
RARE = 0.3                   # un item est rare si son support < 30 %
REPETITIONS = 3              # chaque temps est la médiane de 3 exécutions

FOND = "#fdfcfa"
BLEU, VERT, GRIS = "#2878d6", "#2e9f6e", "#9b9a94"
LARGEUR_BARRE = 0.38


# 1. Lire la sortie de Weka : les lignes sous "Large Itemsets L(k):" sont de la forme "item1 item2 ... nombre".
def lire_itemsets_weka(fichier):
    """Renvoie {frozenset(items): support en nombre de tickets}."""
    frequents = {}
    dans_liste = False
    with open(fichier, encoding="utf-8-sig", errors="replace") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne.startswith("Large Itemsets L("):
                dans_liste = True
                continue
            if not dans_liste:
                continue
            if ligne == "":
                dans_liste = False                 # une ligne vide termine la liste de ce niveau
                continue
            morceaux = ligne.split()
            try:
                support = int(morceaux[-1])
            except ValueError:
                continue
            items = []
            for mot in morceaux[:-1]:
                mot = nettoyer(mot)
                if mot.endswith("=1") or mot.endswith("=True"):
                    mot = mot.rsplit("=", 1)[0]    # colonne 0/1 : l'item s'appelle juste le nom de la colonne
                items.append(mot)
            if any(i.endswith("=0") or i.endswith("=False") for i in items):
                continue                           # main.py ignore aussi les valeurs 0
            frequents[frozenset(items)] = support
    return frequents


# 2. Outils simples.
def seuil_unique(tickets, minsup):
    """Même seuil (en nombre de tickets) pour tous les produits, comme Apriori."""
    s = math.ceil(round(minsup * len(tickets), 9))
    return {p: s for t in tickets for p in t}


def avec_rare(frequents, rares):
    """Nombre d'itemsets qui contiennent au moins un produit rare."""
    return sum(1 for g in frequents if any(p in rares for p in g))


def chrono(f):
    """Exécute f plusieurs fois et renvoie (dernier résultat, temps médian en secondes)."""
    temps = []
    for _ in range(max(1, REPETITIONS)):
        debut = time.perf_counter()
        resultat = f()
        temps.append(time.perf_counter() - debut)
    return resultat, statistics.median(temps)


def pic_memoire(f):
    """Mémoire maximale utilisée pendant f, en Mo."""
    tracemalloc.start()
    f()
    pic = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return pic


def style(ax):
    ax.set_facecolor(FOND)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)


def etiquettes(ax, barres, fmt="%d"):
    for b in barres:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt % b.get_height(),
                ha="center", va="bottom", fontsize=9)


class Rapport:
    """Garde le texte et les graphes, puis les enregistre (PNG, PDF, texte)."""
    def __init__(self, dossier):
        self.dossier = dossier
        self.lignes = []
        self.figures = []
        os.makedirs(dossier, exist_ok=True)

    def ecrire(self, txt=""):
        print(txt)
        self.lignes.append(txt)

    def sauver_figure(self, fig, nom):
        fig.tight_layout()
        fig.savefig(os.path.join(self.dossier, nom), dpi=130)
        self.figures.append(fig)

    def enregistrer(self):
        with PdfPages(os.path.join(self.dossier, "graphes_comparaison.pdf")) as pdf:
            for fig in self.figures:
                pdf.savefig(fig)
        with open(os.path.join(self.dossier, "rapport_comparaison.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lignes))
        print("\nÉcrit dans %s : %d graphes PNG, graphes_comparaison.pdf, rapport_comparaison.txt"
              % (self.dossier, len(self.figures)))


def par_taille(ensemble, tailles):
    return [sum(1 for g in ensemble if len(g) == k) for k in tailles]


# 3. Programme principal.
def main():
    rapport = Rapport(DOSSIER_SORTIE)

    # a) Données et LS.
    tickets = lire_tickets(FICHIER)
    N = len(tickets)
    nb_prod = compter_supports(tickets)
    ls = LS if LS is not None else calculer_ls_auto(nb_prod, N)
    rares = {p for p in nb_prod if nb_prod[p] < RARE * N}
    rapport.ecrire("DONNÉES : %d tickets, %d produits (rares, support < %g %% : %d)"
                   % (N, len(nb_prod), 100 * RARE, len(rares)))
    rapport.ecrire("LS = %.4f (%.2f %%)  -> minsup à donner à Weka = %.4f, soit %d tickets"
                   % (ls, 100 * ls, ls, math.ceil(round(ls * N, 9))))

    # b) Résultats d'Apriori exportés de Weka.
    if not os.path.exists(FICHIER_WEKA):
        print("\nFichier Weka introuvable :", FICHIER_WEKA)
        print("Lance Apriori dans Weka avec le minsup ci-dessus, copie la sortie texte dans ce fichier, puis relance.")
        return
    weka = lire_itemsets_weka(FICHIER_WEKA)
    rapport.ecrire("WEKA : %d itemsets lus dans %s" % (len(weka), os.path.basename(FICHIER_WEKA)))
    if not weka:
        print("Aucun itemset trouvé : as-tu mis outputItemSets = True dans Weka ?")
        return

    # c) CFP-Growth avec le MÊME seuil unique que Weka : on doit retrouver les mêmes itemsets.
    seuils_uniques = seuil_unique(tickets, ls)
    c1, t_unique = chrono(lambda: cfp_growth_seuils(tickets, seuils_uniques))
    cfp1 = {frozenset(g) for g in c1["frequents"]}
    wk = set(weka)
    rapport.ecrire("\n=== PARTIE 1 : Weka (Apriori) et CFP-Growth avec le même seuil unique ===")
    rapport.ecrire("Weka : %d itemsets | CFP-Growth : %d itemsets | identiques : %s"
                   % (len(wk), len(cfp1), "OUI" if wk == cfp1 else "NON"))
    rapport.ecrire("seulement Weka : %d | seulement CFP-Growth : %d" % (len(wk - cfp1), len(cfp1 - wk)))
    if wk != cfp1:
        rapport.ecrire("(petites différences possibles : arrondi du seuil dans Weka, ou noms d'items écrits autrement)")

    tailles = sorted({len(g) for g in wk | cfp1})
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=FOND); style(ax)
    b1 = ax.bar([i - LARGEUR_BARRE / 2 for i in range(len(tailles))], par_taille(wk, tailles), LARGEUR_BARRE,
                color=BLEU, label="Weka Apriori (minsup %.1f %%)" % (100 * ls))
    b2 = ax.bar([i + LARGEUR_BARRE / 2 for i in range(len(tailles))], par_taille(cfp1, tailles), LARGEUR_BARRE,
                color=VERT, label="CFP-Growth (même seuil)")
    etiquettes(ax, b1); etiquettes(ax, b2)
    ax.set_xticks(range(len(tailles))); ax.set_xticklabels(["taille %d" % k for k in tailles])
    ax.set_ylabel("itemsets fréquents"); ax.legend(frameon=False)
    ax.set_title("1. Vérification : mêmes itemsets que Weka ? %s" % ("OUI" if wk == cfp1 else "NON"), loc="left", fontsize=10)
    rapport.sauver_figure(fig, "1_verification_weka_cfp.png")

    # d) mis.py écrit mis.txt, CFP-Growth le lit, puis on compare avec Weka.
    ecrire_mis_txt(calculer_mis(tickets, ls), FICHIER_MIS)
    c2, t_mis = chrono(lambda: cfp_growth(tickets, FICHIER_MIS))
    cfp2 = {frozenset(g) for g in c2["frequents"]}
    rapport.ecrire("\n=== PARTIE 2 : Weka (1 seuil pour tous) contre CFP-Growth (un MIS par produit, lu dans mis.txt) ===")
    rapport.ecrire("Weka : %d itemsets | CFP-Growth : %d itemsets" % (len(wk), len(cfp2)))
    rapport.ecrire("en commun : %d | seulement Weka : %d | seulement CFP-Growth : %d"
                   % (len(wk & cfp2), len(wk - cfp2), len(cfp2 - wk)))
    rapport.ecrire("itemsets avec un produit rare (support < %g %%) : Weka %d | CFP-Growth %d"
                   % (100 * RARE, avec_rare(wk, rares), avec_rare(cfp2, rares)))

    tailles = sorted({len(g) for g in wk | cfp2})
    commun = par_taille(wk & cfp2, tailles)
    seul_w = par_taille(wk - cfp2, tailles)
    seul_c = par_taille(cfp2 - wk, tailles)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=FOND); style(ax)
    ax.bar(range(len(tailles)), commun, color=GRIS, label="en commun")
    ax.bar(range(len(tailles)), seul_w, bottom=commun, color=BLEU, label="seulement Weka")
    ax.bar(range(len(tailles)), seul_c, bottom=[x + y for x, y in zip(commun, seul_w)], color=VERT,
           label="seulement CFP-Growth")
    ax.set_xticks(range(len(tailles))); ax.set_xticklabels(["taille %d" % k for k in tailles])
    ax.set_ylabel("itemsets fréquents"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("2. Weka (1 seuil) contre CFP-Growth (MIS automatique)", loc="left", fontsize=10)
    rapport.sauver_figure(fig, "2_itemsets_communs_et_differents.png")

    # e) Temps et mémoire.
    mem_unique = pic_memoire(lambda: cfp_growth_seuils(tickets, seuils_uniques))
    mem_mis = pic_memoire(lambda: cfp_growth(tickets, FICHIER_MIS))
    rapport.ecrire("\n=== PARTIE 3 : temps et mémoire de CFP-Growth ===")
    rapport.ecrire("CFP-Growth seuil unique : %.2f s, %.1f Mo | CFP-Growth MIS : %.2f s, %.1f Mo"
                   % (t_unique, mem_unique, t_mis, mem_mis))
    if TEMPS_WEKA is not None:
        rapport.ecrire("Weka Apriori : %.2f s (mesuré à la main)" % TEMPS_WEKA)
    rapport.ecrire("MIS-tree : %d produits dans les tickets -> %d nœuds (÷ %.1f)"
                   % (c2["occurrences"], c2["noeuds_arbre"], c2["occurrences"] / max(c2["noeuds_arbre"], 1)))

    noms = ["CFP\nseuil unique", "CFP\nMIS"]
    valeurs = [t_unique, t_mis]
    couleurs = [VERT, VERT]
    if TEMPS_WEKA is not None:
        noms.insert(0, "Weka\nApriori"); valeurs.insert(0, TEMPS_WEKA); couleurs.insert(0, BLEU)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND); style(ax)
    barres = ax.bar(noms, valeurs, color=couleurs, width=0.5)
    etiquettes(ax, barres, "%.2f")
    ax.set_ylabel("temps (s)")
    ax.set_title("3. Temps d'exécution", loc="left", fontsize=10)
    rapport.sauver_figure(fig, "3_temps.png")

    rapport.enregistrer()


if __name__ == "__main__":
    main()