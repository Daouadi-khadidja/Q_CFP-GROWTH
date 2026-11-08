# comparaison_mis.py : compare CFP-Growth avec le MIS CALCULÉ (nos formules) et avec des MIS de comparaison.
# Même moteur CFP-Growth pour toutes les versions : seul le fichier des seuils MIS change. (python comparaison_mis.py)
#
# Versions comparées :
#   1. MIS calculés (nos formules)                              <- notre méthode
#   2. MIS aléatoires entre MIS_MIN et MIS_MAX (12 % - 60 %)    <- comparaison d'origine
#   3. MIS aléatoires dans la MÊME plage que les MIS calculés   <- comparaison équitable (même niveau de seuils)
#
# Les versions 2 et 3 sont tirées avec plusieurs graines : un seul tirage ne prouve rien.
# Graphiques produits (un message par graphique) :
#   0_seuils_mis.png            : les seuils MIS item par item (pourquoi les résultats diffèrent)
#   1_itemsets_et_regles.png    : combien d'itemsets et de règles
#   2_confiance_et_lift.png     : qualité des K meilleures règles (même nombre pour toutes les versions)
#   3_regles_interessantes.png  : part des règles très sûres / très informatives / sur la classe / avec item moins fréquent
import math
import os
import random
import statistics
import textwrap
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cfp_growth import TropDeMotifs, cfp_growth
from main import (CONCLUSION_CONTIENT, CONCLUSION_MAX, CONFIANCE_MIN, FICHIER, LIFT_MIN, SEPARATEUR,
                  lire_tickets)
from mis import SEP, calculer_ls_auto, calculer_mis, compter_supports, ecrire_mis_txt, lire_mis_txt
from mis_aleatoire import MIS_MAX, MIS_MIN, ecrire_mis_aleatoire
from regles import generer_regles, preparer_bitsets

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# RÉGLAGES : les seules valeurs à modifier avant de relancer.
DOSSIER_SORTIE = os.path.join(DOSSIER, "resultats_comparaison_mis")

LS = None                          # None = LS automatique (comme main.py)
GRAINES = [42, 1, 2, 3, 4]         # un tirage par graine
LIMITE_MOTIFS = 3_000_000          # arrêt de sécurité si CFP-Growth énumère trop de groupes (None = pas de limite)
BASES_EQUITABLES = True            # True = ajoute la version 3 ; False = seulement 1 et 2
K_REGLES = None                    # None = K automatique (plus petit nombre de règles) ; sinon un nombre (ex. 100)

RARE = 0.30                        # item "moins fréquent" si son support < 30 %
CONF_FORTE = 0.9                   # règle "très sûre" si confiance >= 90 %
LIFT_FORT = 1.5                    # règle "très informative" si lift >= 1,5
PREFIXE_CLASSE = "label="          # règles qui concluent sur la classe (label=F0L, F1L...)

FOND = "#fdfcfa"
VERT, GRIS, ORANGE = "#2e9f6e", "#9b9a94", "#e08a2c"


def fr(x):
    """Nombre avec virgule décimale (1.5 -> 1,5)."""
    return ("%g" % x).replace(".", ",")


# 1. Lance CFP-Growth (il lit le fichier MIS) puis génère les règles, et mesure ce qu'on veut comparer.
def evaluer(nom, fichier_mis, tickets, bitsets, N, rares, rapport, mesurer_memoire=False):
    rapport.ecrire("\n--- %s ---" % nom)

    # Temps : mesuré SANS tracemalloc (qui ralentit beaucoup l'exécution).
    debut = time.perf_counter()
    try:
        res = cfp_growth(tickets, fichier_mis, limite_motifs=LIMITE_MOTIFS)
    except TropDeMotifs as e:
        rapport.ecrire("ARRÊT : trop de groupes énumérés (%s). Ce MIS est trop bas pour ces données." % e)
        return None
    temps = time.perf_counter() - debut

    # Mémoire : mesurée dans une seconde exécution (seulement pour le premier tirage de chaque version).
    memoire = None
    if mesurer_memoire:
        tracemalloc.start()
        cfp_growth(tickets, fichier_mis, limite_motifs=LIMITE_MOTIFS)
        memoire = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()

    frequents = res["frequents"]
    regles = generer_regles(frequents, N, bitsets, confiance_min=CONFIANCE_MIN, lift_min=LIFT_MIN,
                            conclusion_max=CONCLUSION_MAX, conclusion_contient=CONCLUSION_CONTIENT,
                            separateur=SEPARATEUR)      # déjà triées par lift décroissant

    confs = [r["confiance"] for r in regles]
    lifts = [r["lift"] for r in regles]
    n = len(regles)

    def moy(x):
        return statistics.mean(x) if x else 0.0

    def pct(nombre):
        return 100.0 * nombre / n if n else 0.0

    def avec_rare(r):
        return any(i in rares for i in (r["si"].split(SEPARATEUR) + r["alors"].split(SEPARATEUR)))

    m = {
        "nom": nom,
        "itemsets": len(frequents),
        "regles": n,
        "conf_moy": moy(confs), "lift_moy": moy(lifts),
        "pct_sures": pct(sum(1 for c in confs if c >= CONF_FORTE)),
        "pct_lift_fort": pct(sum(1 for l in lifts if l >= LIFT_FORT)),
        "pct_classe": pct(sum(1 for r in regles if r["alors"].startswith(PREFIXE_CLASSE))),
        "pct_rares": pct(sum(1 for r in regles if avec_rare(r))),
        "temps": temps, "memoire": memoire,
        "confs": confs, "lifts": lifts,
        "cles": {(frozenset(r["si"].split(SEPARATEUR)), r["alors"]) for r in regles},
        "top": regles[:5],
    }
    rapport.ecrire("%d itemsets, %d règles, %.1f s%s"
                   % (m["itemsets"], n, temps, (", %.0f Mo" % memoire) if memoire is not None else ""))
    return m


# 3. Rapport : garde le texte et les graphes puis les enregistre.
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
        fig.savefig(os.path.join(self.dossier, nom), dpi=150, facecolor=fig.get_facecolor())
        self.figures.append(fig)

    def enregistrer(self):
        with PdfPages(os.path.join(self.dossier, "graphes_comparaison_mis.pdf")) as pdf:
            for fig in self.figures:
                pdf.savefig(fig)
        with open(os.path.join(self.dossier, "rapport_comparaison_mis.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lignes))
        print("\nÉcrit dans %s : %d graphes PNG, graphes_comparaison_mis.pdf, rapport_comparaison_mis.txt"
              % (self.dossier, len(self.figures)))


# 4. Outils de dessin : un style simple, des barres avec la valeur écrite dessus, un point noir par tirage.
def style(ax):
    ax.set_facecolor(FOND)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)


def barres(ax, groupes, cle, fmt="%.0f", titre="", ylabel="", pourcentage=False, log_auto=False,
           court=False, ligne=None):
    """Une barre par version = moyenne des tirages. Un point noir par tirage. La valeur est écrite au-dessus."""
    style(ax)
    valeurs = [[r[cle] for r in g["runs"]] for g in groupes]
    tout = [v for vals in valeurs for v in vals]
    log = log_auto and min(tout) > 0 and max(tout) / min(tout) > 30
    if log:
        ax.set_yscale("log")
    for i, (g, vals) in enumerate(zip(groupes, valeurs)):
        moyenne = statistics.mean(vals)
        ax.bar(i, moyenne, width=0.62, color=g["couleur"], zorder=2)
        if len(vals) > 1:
            xs = [i - 0.17 + 0.34 * k / (len(vals) - 1) for k in range(len(vals))]
            ax.scatter(xs, vals, s=16, color="black", zorder=4)
        ax.annotate(fmt % moyenne, (i, max([moyenne] + vals)), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(groupes)))
    ax.set_xticklabels([g["court"] if court else g["nom"] for g in groupes], fontsize=7.5 if court else 8)
    ax.set_title(titre + (" (échelle log)" if log else ""), loc="left", fontsize=10)
    ax.set_ylabel(ylabel)
    if pourcentage:
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
    elif not log:
        ax.set_ylim(0, max(tout) * 1.18 if max(tout) > 0 else 1)
    if ligne is not None:                       # ligne de référence (ex. lift = 1 : pas de lien)
        ax.axhline(ligne, color="#555555", ls="--", lw=1, zorder=3)


def titre_et_note(fig, titre, note):
    fig.suptitle(titre, x=0.01, ha="left", fontweight="bold", fontsize=12)
    note = textwrap.fill(note, 170)
    fig.text(0.01, 0.012, note, fontsize=8, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04 + 0.025 * note.count("\n"), 1, 0.93))


NOTE_LECTURE = ("Lecture : chaque barre est la moyenne de la version ; chaque point noir est un tirage "
                "aléatoire (une graine). Vert = notre méthode.")


# 5. Programme principal.
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

    m = calculer_mis(tickets, ls)
    mis_bas, mis_haut = min(m["mis"].values()), max(m["mis"].values())
    plage_calc = "%s–%s %%" % (fr(round(100 * mis_bas, 1)), fr(round(100 * mis_haut, 1)))
    plage_alea = "%s–%s %%" % (fr(round(100 * MIS_MIN)), fr(round(100 * MIS_MAX)))

    rapport.ecrire("DONNÉES : %d tickets, %d items (moins fréquents, support < %g %% : %d)"
                   % (N, len(nb), 100 * RARE, len(rares)))
    rapport.ecrire("RÈGLES : confiance >= %g %%, lift > %g, conclusion de %d item(s) au plus"
                   % (100 * CONFIANCE_MIN, LIFT_MIN, CONCLUSION_MAX))
    rapport.ecrire("MIS calculés : LS = %.2f %%, MIS entre %s | MIS aléatoires d'origine : entre %s | graines %s"
                   % (100 * ls, plage_calc, plage_alea, GRAINES))

    fichier_calcule = os.path.join(DOSSIER_SORTIE, "mis_calcule.txt")
    ecrire_mis_txt(m, fichier_calcule)
    produits = [(p, m["support_rel"][p]) for p in sorted(m["mis"], key=lambda p: (-m["mis"][p], p))]

    # a) Lancer chaque version (plusieurs graines pour les versions tirées au hasard).
    def lancer(cle, nom, nom_court, couleur, fabriquer, graines, court=""):
        runs = []
        for i, graine in enumerate(graines):
            fichier = os.path.join(DOSSIER_SORTIE, "mis_%s_graine%d.txt" % (cle, graine)) if graine is not None \
                else fichier_calcule
            if fabriquer is not None:
                fabriquer(fichier, graine)
            titre = nom_court if graine is None else "%s, graine %d" % (nom_court, graine)
            r = evaluer(titre, fichier, tickets, bitsets, N, rares, rapport, mesurer_memoire=(i == 0))
            if r is not None:
                runs.append(r)
        return {"cle": cle, "nom": nom, "court": court or nom, "couleur": couleur, "runs": runs}

    groupes = [lancer("calcule", "Q-CFP-Growth\n(MIS calculés\n%s)" % plage_calc, "MIS calculés", VERT, None, [None], court="MIS\ncalculés"),
               lancer("aleatoire", "MIS aléatoires\n%s" % plage_alea, "MIS aléatoires %s" % plage_alea, GRIS,
                      lambda f, g: ecrire_mis_aleatoire(produits, N, f, MIS_MIN, MIS_MAX, g), GRAINES,
                      court="aléat.\n%s" % plage_alea)]
    if BASES_EQUITABLES:
        groupes.append(lancer("aleatoire_equitable", "MIS aléatoires\nmême plage que\nles MIS calculés",
                              "MIS aléatoires %s" % plage_calc, ORANGE,
                              lambda f, g: ecrire_mis_aleatoire(produits, N, f, mis_bas, mis_haut, g), GRAINES,
                              court="aléat.\nmême plage"))

    for g in groupes:
        if not g["runs"]:
            rapport.ecrire("\nVersion « %s » : aucun résultat (arrêt de sécurité), elle est absente des graphes."
                           % g["nom"].replace("\n", " "))
    groupes = [g for g in groupes if g["runs"]]
    if len(groupes) < 2:
        rapport.ecrire("\nPas assez de résultats pour comparer.")
        rapport.enregistrer()
        return
    calcule = next((g for g in groupes if g["cle"] == "calcule"), None)

    # b) Tableau récapitulatif (moyenne des tirages de chaque version).
    rapport.ecrire("\n=== TABLEAU : moyenne des tirages de chaque version ===")
    colonnes = [("itemsets", "itemsets", "%.0f"), ("regles", "règles", "%.0f"),
                ("conf_moy", "conf. moy.", "%.3f"), ("lift_moy", "lift moy.", "%.2f"),
                ("pct_sures", "%% conf>=%g" % (100 * CONF_FORTE), "%.1f"), ("pct_lift_fort", "%% lift>=%g" % LIFT_FORT, "%.1f"),
                ("pct_classe", "% concl. classe", "%.1f"), ("pct_rares", "% item moins fréq.", "%.1f"),
                ("temps", "temps (s)", "%.1f")]
    rapport.ecrire("%-34s %4s " % ("version", "nb") + " ".join("%17s" % c[1] for c in colonnes))
    for g in groupes:
        rapport.ecrire("%-34s %4d " % (g["nom"].replace("\n", " "), len(g["runs"]))
                       + " ".join("%17s" % (c[2] % statistics.mean(r[c[0]] for r in g["runs"])) for c in colonnes))
    for g in groupes:
        mems = [r["memoire"] for r in g["runs"] if r["memoire"] is not None]
        if mems:
            rapport.ecrire("Mémoire maximale (1er tirage) %-34s : %.0f Mo" % (g["nom"].replace("\n", " "), mems[0]))

    # c) K meilleures règles : même nombre de règles pour toutes les versions (comparaison équitable).
    toutes = [r for g in groupes for r in g["runs"]]
    K = K_REGLES if K_REGLES else min(r["regles"] for r in toutes)
    for r in toutes:
        haut = min(K, r["regles"])
        r["topk_lift"] = statistics.mean(r["lifts"][:haut]) if haut else 0.0
        r["topk_conf"] = 100 * statistics.mean(r["confs"][:haut]) if haut else 0.0
    if 0 < K < 10:
        rapport.ecrire("\nATTENTION : K = %d seulement (une version produit très peu de règles) : la comparaison à nombre "
                       "égal est peu fiable. Mets K_REGLES = 20 ou 50 en haut du fichier pour la rendre plus parlante." % K)
    if K > 0:
        rapport.ecrire("\n=== À NOMBRE ÉGAL : les %d meilleures règles (par lift) de chaque tirage ===" % K)
        for g in groupes:
            rapport.ecrire("%-34s confiance moy. = %.1f %% | lift moy. = %.2f"
                           % (g["nom"].replace("\n", " "), statistics.mean(r["topk_conf"] for r in g["runs"]),
                              statistics.mean(r["topk_lift"] for r in g["runs"])))

    # d) Recouvrement avec les règles du MIS calculé, et meilleures règles.
    if calcule is not None:
        ref = calcule["runs"][0]
        rapport.ecrire("\n=== RECOUVREMENT : règles du MIS calculé retrouvées dans les autres versions (moyenne) ===")
        for g in groupes:
            if g is calcule:
                continue
            communs = statistics.mean(len(ref["cles"] & r["cles"]) for r in g["runs"])
            rapport.ecrire("%-34s : %.0f en commun | %.0f seulement calculé | %.0f seulement cette version"
                           % (g["nom"].replace("\n", " "), communs, len(ref["cles"]) - communs,
                              statistics.mean(len(r["cles"] - ref["cles"]) for r in g["runs"])))
        rapport.ecrire("\n=== 5 MEILLEURES RÈGLES (par lift) ===")
        for r in (ref, groupes[1]["runs"][0]):
            rapport.ecrire("\n%s :" % r["nom"])
            for reg in r["top"]:
                rapport.ecrire("   SI %s  ALORS %s | support %.4f | confiance %.3f | lift %.2f"
                               % (reg["si"], reg["alors"], reg["support"], reg["confiance"], reg["lift"]))

    # e) GRAPHIQUES -----------------------------------------------------------------------------------------

    # Graphique 0 : les seuils item par item. C'est LE graphique qui explique pourquoi les versions diffèrent.
    items = sorted(nb, key=lambda p: nb[p])                     # du moins fréquent au plus fréquent
    x = list(range(1, len(items) + 1))
    support_pct = [100 * nb[p] / N for p in items]
    mis_calc_pct = [100 * m["mis"][p] for p in items]
    mis_alea, _ = lire_mis_txt(os.path.join(DOSSIER_SORTIE, "mis_aleatoire_graine%d.txt" % GRAINES[0]))
    mis_alea_pct = [100 * mis_alea[p] for p in items]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor=FOND)
    style(ax)
    ax.scatter(x, mis_alea_pct, s=14, color=GRIS, zorder=2, label="MIS aléatoire (%s, graine %d)" % (plage_alea, GRAINES[0]))
    ax.plot(x, mis_calc_pct, color=VERT, lw=2.2, zorder=3, label="MIS calculé (notre méthode)")
    ax.plot(x, support_pct, color="black", lw=1.4, zorder=4, label="support réel de l'item")
    ax.axhline(100 * mis_bas, color=VERT, ls="--", lw=1)
    ax.annotate("MIN = plus petit MIS calculé (%s %%) : en dessous, l'item est éliminé d'emblée" % fr(round(100 * mis_bas, 1)),
                (1, 100 * mis_bas), xytext=(4, 5), textcoords="offset points", fontsize=8, color=VERT)
    ax.set_xlabel("items, du moins fréquent (gauche) au plus fréquent (droite)")
    ax.set_ylabel("support ou seuil MIS (%)")
    ax.set_ylim(0, min(100, 1.08 * max(max(support_pct), max(mis_alea_pct))))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    titre_et_note(fig, "0. Les seuils MIS item par item : calculés (vert) et tirés au hasard (gris)",
                  "Lecture : un item dont le support (ligne noire) est sous son seuil ne peut pas être fréquent seul. "
                  "Les seuils aléatoires sont indépendants du support ; les seuils calculés en dépendent (plancher LS, puis bêta × support).")
    rapport.sauver_figure(fig, "0_seuils_mis.png")

    # Graphique 1 : combien d'itemsets et de règles.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=FOND)
    barres(ax1, groupes, "itemsets", "%.0f", "(a) Itemsets fréquents trouvés", log_auto=True)
    barres(ax2, groupes, "regles", "%.0f", "(b) Règles d'association retenues", log_auto=True)
    titre_et_note(fig, "1. Combien d'itemsets et de règles chaque version trouve-t-elle ?", NOTE_LECTURE)
    rapport.sauver_figure(fig, "1_itemsets_et_regles.png")

    # Graphique 2 : qualité des K meilleures règles (même nombre pour tout le monde).
    if K > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=FOND)
        barres(ax1, groupes, "topk_conf", "%.0f %%", "(a) Confiance moyenne", ylabel="confiance (%)", pourcentage=True)
        barres(ax2, groupes, "topk_lift", "%.2f", "(b) Lift moyen", ylabel="lift (pointillé = 1 : pas de lien)", ligne=1)
        titre_et_note(fig, "2. Qualité des %d meilleures règles de chaque version (même nombre pour toutes)" % K,
                      NOTE_LECTURE + " On compare le même nombre de règles (les meilleures par lift) pour ne pas "
                      "favoriser la version qui en produit le plus."
                      + (" K est petit : à interpréter avec prudence." if K < 10 else ""))
        rapport.sauver_figure(fig, "2_confiance_et_lift.png")
    else:
        rapport.ecrire("\nGraphique 2 non tracé : une version n'a aucune règle (K = 0).")

    # Graphique 3 : quelle part des règles est intéressante ? (en %, pour comparer des versions de tailles différentes)
    criteres = [("pct_sures", "… très sûres\n(confiance ≥ %s %%)" % fr(100 * CONF_FORTE)),
                ("pct_lift_fort", "… très informatives\n(lift ≥ %s)" % fr(LIFT_FORT)),
                ("pct_classe", "… qui prédisent la classe\n(conclusion %s…)" % PREFIXE_CLASSE),
                ("pct_rares", "… avec un item moins fréquent\n(support < %s %%)" % fr(100 * RARE))]
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.2), facecolor=FOND)
    for ax, (cle, titre) in zip(axes, criteres):
        barres(ax, groupes, cle, "%.0f %%", "Part de règles " + titre, pourcentage=True, court=True)
    axes[0].set_ylabel("part des règles de la version (%)")
    titre_et_note(fig, "3. Quelle part des règles est intéressante ?",
                  NOTE_LECTURE + " Les parts sont en % des règles de chaque version, pour comparer des versions qui "
                  "ne produisent pas le même nombre de règles.")
    rapport.sauver_figure(fig, "3_regles_interessantes.png")

    rapport.enregistrer()


if __name__ == "__main__":
    main()