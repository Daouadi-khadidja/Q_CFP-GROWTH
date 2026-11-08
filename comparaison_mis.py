# comparaison_mis.py : compare CFP-Growth avec le MIS CALCULÉ (nos formules) et avec des MIS ALÉATOIRES.
# Pour chaque version : itemsets fréquents, règles d'association, et qualité des règles. (python comparaison_mis.py)
#
# Idée : si nos formules sont utiles, les règles obtenues avec le MIS calculé doivent être meilleures
# (ou plus intéressantes) que celles obtenues avec des MIS tirés au hasard.
# On teste PLUSIEURS tirages aléatoires (graines) : un seul tirage ne prouve rien, il peut être chanceux ou malchanceux.
import os
import statistics
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cfp_growth import TropDeMotifs, cfp_growth
from main import (CONCLUSION_CONTIENT, CONCLUSION_MAX, CONFIANCE_MIN, FICHIER, LIFT_MIN, SEPARATEUR,
                  lire_tickets)
from mis import calculer_ls_auto, calculer_mis, compter_supports, ecrire_mis_txt
from mis_aleatoire import MIS_MAX, MIS_MIN, ecrire_mis_aleatoire
from regles import generer_regles, preparer_bitsets

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# RÉGLAGES : les seules valeurs à modifier avant de relancer.
DOSSIER_SORTIE = os.path.join(DOSSIER, "resultats_comparaison_mis")

LS = None                          # None = LS automatique (comme main.py)
GRAINES = [42, 1, 2, 3, 4]         # un tirage aléatoire par graine (42 = celui de mis_aleatoire.txt)
LIMITE_MOTIFS = 3_000_000          # arrêt de sécurité si CFP-Growth énumère trop de groupes (None = pas de limite)

RARE = 0.05                        # un item est rare si son support < 5 %
CONF_FORTE = 0.9                   # une règle est "très sûre" si sa confiance >= 90 %
LIFT_FORT = 1.5                    # une règle est "très informative" si son lift >= 1,5
PREFIXE_CLASSE = "label="          # les règles qui concluent sur la classe (label=F0L, F1L...) sont les plus utiles

FOND = "#fdfcfa"
VERT, GRIS = "#2e9f6e", "#9b9a94"


# 1. Lance CFP-Growth (il lit le fichier MIS) puis génère les règles, et mesure ce qu'on veut comparer.
def evaluer(nom, fichier_mis, tickets, bitsets, N, rares, rapport):
    rapport.ecrire("\n--- %s ---" % nom)
    debut = time.perf_counter()
    tracemalloc.start()
    try:
        res = cfp_growth(tickets, fichier_mis, limite_motifs=LIMITE_MOTIFS)
    except TropDeMotifs as e:
        tracemalloc.stop()
        rapport.ecrire("ARRÊT : trop de groupes énumérés (%s). Ce MIS est trop bas pour ces données." % e)
        return None
    pic = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    temps_cfp = time.perf_counter() - debut

    frequents = res["frequents"]
    regles = generer_regles(frequents, N, bitsets, confiance_min=CONFIANCE_MIN, lift_min=LIFT_MIN,
                            conclusion_max=CONCLUSION_MAX, conclusion_contient=CONCLUSION_CONTIENT,
                            separateur=SEPARATEUR)

    confs = [r["confiance"] for r in regles]
    lifts = [r["lift"] for r in regles]
    sups = [r["support"] for r in regles]

    def avec_rare(r):
        return any(i in rares for i in (r["si"].split(SEPARATEUR) + r["alors"].split(SEPARATEUR)))

    def moy(x):
        return statistics.mean(x) if x else 0.0

    def med(x):
        return statistics.median(x) if x else 0.0

    m = {
        "nom": nom,
        "itemsets": len(frequents),
        "itemsets_rares": sum(1 for g in frequents if any(p in rares for p in g)),
        "regles": len(regles),
        "conf_moy": moy(confs), "conf_med": med(confs),
        "lift_moy": moy(lifts), "lift_med": med(lifts),
        "sup_moy": moy(sups),
        "regles_rares": sum(1 for r in regles if avec_rare(r)),
        "regles_classe": sum(1 for r in regles if r["alors"].startswith(PREFIXE_CLASSE)),
        "regles_sures": sum(1 for c in confs if c >= CONF_FORTE),
        "regles_lift_fort": sum(1 for l in lifts if l >= LIFT_FORT),
        "temps": temps_cfp, "memoire": pic,
        "confs": confs, "lifts": lifts,
        "cles": {(frozenset(r["si"].split(SEPARATEUR)), r["alors"]) for r in regles},
        "top": regles[:5],
    }
    rapport.ecrire("%d itemsets (dont %d avec un item rare), %d règles, %.1f s, %.0f Mo"
                   % (m["itemsets"], m["itemsets_rares"], m["regles"], temps_cfp, pic))
    return m


# 2. Rapport : garde le texte et les graphes puis les enregistre.
class Rapport:
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
        with PdfPages(os.path.join(self.dossier, "graphes_comparaison_mis.pdf")) as pdf:
            for fig in self.figures:
                pdf.savefig(fig)
        with open(os.path.join(self.dossier, "rapport_comparaison_mis.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lignes))
        print("\nÉcrit dans %s : %d graphes PNG, graphes_comparaison_mis.pdf, rapport_comparaison_mis.txt"
              % (self.dossier, len(self.figures)))


def style(ax):
    ax.set_facecolor(FOND)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)


def etiquettes(ax, barres, fmt="%d"):
    for b in barres:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt % b.get_height(),
                ha="center", va="bottom", fontsize=8)


def noms_courts(mesures):
    return ["MIS\ncalculé" if m["nom"].startswith("MIS calculé") else "aléat.\n%s" % m["nom"].split("graine ")[-1]
            for m in mesures]


# 3. Programme principal.
def main():
    rapport = Rapport(DOSSIER_SORTIE)

    if not os.path.exists(FICHIER):
        print("Fichier introuvable :", FICHIER)
        print("Lance d'abord : python nettoyer.py")
        return

    tickets = lire_tickets(FICHIER)
    N = len(tickets)
    nb = compter_supports(tickets)
    ls = LS if LS is not None else calculer_ls_auto(nb, N)
    rares = {p for p in nb if nb[p] < RARE * N}
    bitsets = preparer_bitsets(tickets)
    rapport.ecrire("DONNÉES : %d tickets, %d items (rares, support < %g %% : %d)"
                   % (N, len(nb), 100 * RARE, len(rares)))
    rapport.ecrire("RÈGLES : confiance >= %g %%, lift > %g, conclusion de %d item(s) au plus"
                   % (100 * CONFIANCE_MIN, LIFT_MIN, CONCLUSION_MAX))
    rapport.ecrire("MIS calculé : LS = %.2f %% | MIS aléatoires : tirés entre %.0f %% et %.0f %%, graines %s"
                   % (100 * ls, 100 * MIS_MIN, 100 * MIS_MAX, GRAINES))

    # a) MIS calculé par nos formules (mis.py) : écrit dans un fichier, puis lu par CFP-Growth.
    m = calculer_mis(tickets, ls)
    fichier_calcule = os.path.join(DOSSIER_SORTIE, "mis_calcule.txt")
    ecrire_mis_txt(m, fichier_calcule)

    mesures = []
    r = evaluer("MIS calculé (formules)", fichier_calcule, tickets, bitsets, N, rares, rapport)
    if r is not None:
        mesures.append(r)

    # b) MIS aléatoires : mêmes produits (même ordre que mis.txt), valeurs tirées au hasard.
    produits = [(p, m["support_rel"][p]) for p in sorted(m["mis"], key=lambda p: (-m["mis"][p], p))]
    for graine in GRAINES:
        fichier = os.path.join(DOSSIER_SORTIE, "mis_aleatoire_graine%d.txt" % graine)
        ecrire_mis_aleatoire(produits, N, fichier, MIS_MIN, MIS_MAX, graine)
        r = evaluer("MIS aléatoire, graine %d" % graine, fichier, tickets, bitsets, N, rares, rapport)
        if r is not None:
            mesures.append(r)

    if len(mesures) < 2:
        rapport.ecrire("\nPas assez de résultats pour comparer.")
        rapport.enregistrer()
        return

    calcule = mesures[0]
    aleatoires = mesures[1:]

    # c) Tableau récapitulatif.
    rapport.ecrire("\n=== TABLEAU : qualité des règles ===")
    colonnes = [("itemsets", "itemsets", "%d"), ("regles", "règles", "%d"),
                ("conf_moy", "conf. moy.", "%.3f"), ("lift_moy", "lift moy.", "%.2f"), ("lift_med", "lift méd.", "%.2f"),
                ("regles_sures", "conf>=%g" % CONF_FORTE, "%d"), ("regles_lift_fort", "lift>=%g" % LIFT_FORT, "%d"),
                ("regles_rares", "avec item rare", "%d"), ("regles_classe", "concl. %s" % PREFIXE_CLASSE, "%d"),
                ("temps", "temps (s)", "%.1f")]
    rapport.ecrire("%-26s " % "version" + " ".join("%14s" % c[1] for c in colonnes))
    for x in mesures:
        rapport.ecrire("%-26s " % x["nom"] + " ".join("%14s" % (c[2] % x[c[0]]) for c in colonnes))
    if len(aleatoires) > 1:
        rapport.ecrire("%-26s " % "moyenne des aléatoires"
                       + " ".join("%14s" % (c[2] % statistics.mean(x[c[0]] for x in aleatoires)) for c in colonnes))

    # Comparaison à nombre égal : un MIS plus bas donne plus de règles, ce qui peut tromper sur les moyennes.
    # On compare donc aussi les K meilleures règles de chaque version (K = le plus petit nombre de règles).
    K = min(x["regles"] for x in mesures)
    if K > 0:
        rapport.ecrire("\nÀ nombre égal (les %d meilleures règles par lift de chaque version) :" % K)
        for x in mesures:
            meilleurs = sorted(x["lifts"], reverse=True)[:K]
            rapport.ecrire("%-26s lift moyen = %.2f" % (x["nom"], statistics.mean(meilleurs)))

    # d) Recouvrement des règles et meilleures règles.
    rapport.ecrire("\n=== RECOUVREMENT : règles du MIS calculé retrouvées avec un MIS aléatoire ===")
    for x in aleatoires:
        commun = len(calcule["cles"] & x["cles"])
        rapport.ecrire("%-26s : %d en commun | %d seulement calculé | %d seulement aléatoire"
                       % (x["nom"], commun, len(calcule["cles"] - x["cles"]), len(x["cles"] - calcule["cles"])))

    rapport.ecrire("\n=== 5 MEILLEURES RÈGLES (par lift) ===")
    for x in (calcule, aleatoires[0]):
        rapport.ecrire("\n%s :" % x["nom"])
        for reg in x["top"]:
            rapport.ecrire("   SI %s  ALORS %s | support %.4f | confiance %.3f | lift %.2f"
                           % (reg["si"], reg["alors"], reg["support"], reg["confiance"], reg["lift"]))

    # e) Graphes.
    noms = noms_courts(mesures)
    couleurs = [VERT] + [GRIS] * len(aleatoires)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=FOND); style(ax1); style(ax2)
    etiquettes(ax1, ax1.bar(noms, [x["itemsets"] for x in mesures], color=couleurs))
    ax1.set_title("(a) Itemsets fréquents", loc="left", fontsize=10)
    etiquettes(ax2, ax2.bar(noms, [x["regles"] for x in mesures], color=couleurs))
    ax2.set_title("(b) Règles d'association", loc="left", fontsize=10)
    fig.suptitle("1. Combien d'itemsets et de règles ?  (vert = MIS calculé, gris = MIS aléatoire)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    rapport.sauver_figure(fig, "1_itemsets_et_regles.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=FOND); style(ax1); style(ax2)
    boite1 = ax1.boxplot([x["confs"] for x in mesures], patch_artist=True, showfliers=False)
    boite2 = ax2.boxplot([x["lifts"] for x in mesures], patch_artist=True, showfliers=False)
    for boite in (boite1, boite2):
        for case, c in zip(boite["boxes"], couleurs):
            case.set_facecolor(c)
    for ax in (ax1, ax2):
        ax.set_xticks(range(1, len(noms) + 1)); ax.set_xticklabels(noms, fontsize=8)
    ax1.set_ylabel("confiance"); ax1.set_title("(a) Confiance des règles", loc="left", fontsize=10)
    ax2.set_ylabel("lift"); ax2.set_title("(b) Lift des règles", loc="left", fontsize=10)
    fig.suptitle("2. Qualité des règles (la boîte contient la moitié centrale des règles)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    rapport.sauver_figure(fig, "2_confiance_et_lift.png")

    fig, axes = plt.subplots(1, 4, figsize=(13, 4), facecolor=FOND)
    criteres = [("regles_sures", "confiance >= %g" % CONF_FORTE), ("regles_lift_fort", "lift >= %g" % LIFT_FORT),
                ("regles_rares", "avec un item rare"), ("regles_classe", "concluent sur %s" % PREFIXE_CLASSE)]
    for ax, (cle, titre) in zip(axes, criteres):
        style(ax)
        etiquettes(ax, ax.bar(noms, [x[cle] for x in mesures], color=couleurs))
        ax.set_title(titre, loc="left", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("3. Règles intéressantes : combien en trouve-t-on ?", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    rapport.sauver_figure(fig, "3_regles_interessantes.png")

    rapport.enregistrer()


if __name__ == "__main__":
    main()
