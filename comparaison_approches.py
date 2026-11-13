# comparaison_approches.py : compare notre approche (MIS calculés automatiquement) avec CFP-Growth original.
# Même moteur CFP-Growth pour toutes les versions : seule la façon d'attribuer les MIS change. (python comparaison_approches.py)
#
# Versions comparées :
#   NOTRE APPROCHE  : MIS_i = bêta(catégorie) x support_i, bêtas calculés automatiquement.
#   ORIGINALE       : MIS_i = max(β x support_i, LS), avec des valeurs de β et LS choisies à la main (expérimentales).
#   ALÉATOIRE       : MIS_i tiré au hasard (plage large, puis même plage que notre approche), plusieurs graines.
#   RÉFÉRENCE       : seuil unique de 1 ticket (sert seulement à mesurer quelles règles sont perdues).
#
# Graphiques (un message par graphique, barres horizontales, vert = notre approche) :
#   1. Seuil minimum médian par type d'item (faible / moyenne / forte)
#   2. Part des items rares conservés
#   3. Part des règles fiables retrouvées (par rapport à la référence)
#   4. Part de règles fiables parmi toutes les règles produites
#   5. Qualité des règles fiables : confiance et lift
#   6. Temps de calcul
#   7. Effet des réglages β et LS de la formule originale
#   8. Tableau de synthèse
import csv
import math
import os
import random
import statistics
import textwrap
import time
from collections import defaultdict
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cfp_growth import Noeud, TropDeMotifs, inserer_chemin, fouiller_arbre
from main import CONCLUSION_MAX, CONFIANCE_MIN, FICHIER, LIFT_MIN, SEPARATEUR, lire_tickets
from mis import calculer_mis_categories, categories_quartiles, compter_supports
from mis_aleatoire import MIS_MAX, MIS_MIN
from regles import generer_regles, preparer_bitsets

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# RÉGLAGES : les seules valeurs à modifier avant de relancer.
DOSSIER_SORTIE = os.path.join(DOSSIER, "resultats_comparaison_approches")

TAILLE_MAX = 3                 # taille maximale des itemsets (3 = rapide ; 4 = plus long)
LIMITE_MOTIFS = 3_000_000      # arrêt de sécurité si trop de groupes sont énumérés
REPETITIONS = 3                # chaque temps est la médiane de 3 exécutions

RARE = 0.05                    # un item est rare si son support < 5 %
MIN_FIABLE = 5                 # une règle est fiable si elle repose sur au moins 5 tickets

# Formule originale max(β x support, LS) : LS est donné en nombre de tickets.
BETAS_BALAYAGE = [0.0002, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]   # β varie, LS fixé
LS_FIXE = 1                                                                      # LS (tickets) pendant le balayage de β
LS_BALAYAGE = [1, 5, 11, 55, 110, 547]                                           # LS varie (tickets), β fixé
BETA_FIXE = 0.05                                                                 # β pendant le balayage de LS
ORIGINALES = [(0.05, 1), (0.5, 1), (0.1, 110), (0.2, 547)]                       # réglages (β, LS tickets) comparés partout

GRAINES = [42, 1, 2, 3, 4]     # un tirage aléatoire par graine
AVEC_REFERENCE = True          # True = mesure aussi les règles perdues par rapport au seuil unique de 1 ticket

# Couleurs : une couleur par famille de versions, toujours la même dans tous les graphes.
FOND = "#fdfcfa"
VERT, BLEU, GRIS, ORANGE = "#2e9f6e", "#2878d6", "#9b9a94", "#e08a2c"
BLEU_FONCE = "#1c5aa6"
ENCRE, ENCRE_2 = "#222222", "#555555"
NOMS_CAT = ["faible", "moyenne", "forte"]


# Écrit un nombre avec une virgule décimale (x = 0,05).
def fr(x):
    return ("%g" % x).replace(".", ",")


# Écrit un nombre avec une virgule décimale et un nombre fixe de décimales (x = 12,5).
def mf(x, decimales=0):
    return ("%.*f" % (decimales, x)).replace(".", ",")


# 1. Attribution des MIS (chaque fonction renvoie (seuils en tickets, MIS en proportion)).
def seuils_depuis_mis(mis, N):
    return {p: max(1, math.ceil(round(v * N, 9))) for p, v in mis.items()}


# Notre approche : calculée automatiquement par mis.py.
def mis_notre(m):
    return m["seuils"], m["mis"]


# CFP-Growth original : MIS = max(β x support, LS), avec β et LS choisis à la main.
def mis_originale(nb, N, beta, ls_tickets):
    mis = {p: max(beta * nb[p] / N, ls_tickets / N) for p in nb}
    return seuils_depuis_mis(mis, N), mis


# Aléatoire : MIS tiré au hasard entre bas et haut.
def mis_aleatoire(nb, N, bas, haut, graine):
    hasard = random.Random(graine)
    mis = {p: hasard.uniform(bas, haut) for p in sorted(nb)}
    return seuils_depuis_mis(mis, N), mis


# Seuil unique (référence).
def mis_unique(nb, N, tickets_min):
    return {p: tickets_min for p in nb}, {p: tickets_min / N for p in nb}


# 2. CFP-Growth : même code que cfp_growth.py, mais compte aussi les candidats énumérés.
def cfp(D, seuils, mis_tri):
    MIN = min(seuils[p] for p in D.nb)
    ordre = sorted((p for p in D.nb if D.nb[p] >= MIN), key=lambda p: (-mis_tri[p], -D.nb[p], p))
    rang = {p: i for i, p in enumerate(ordre)}

    racine = Noeud(None, None)
    entete = defaultdict(list)
    stats = {"noeuds": 0, "bases": 0, "limite": LIMITE_MOTIFS, "taille_max": TAILLE_MAX}
    for ticket in D.ensembles:
        chemin = sorted((p for p in ticket if p in rang), key=lambda p: rang[p])
        if chemin:
            inserer_chemin(racine, chemin, 1, entete, stats)

    brut = {}
    fouiller_arbre(entete, rang, (), MIN, brut, stats)
    frequents = {tuple(sorted(g)): s for g, s in brut.items() if s >= min(seuils[p] for p in g)}
    return frequents, len(brut), stats["noeuds"]


# 3. Évalue une version : itemsets, candidats, temps, règles, fiabilité, items rares.
def evaluer(nom, famille, seuils, mis, D):
    temps = []
    try:
        for _ in range(REPETITIONS):
            debut = time.perf_counter()
            frequents, candidats, noeuds = cfp(D, seuils, mis)
            temps.append(time.perf_counter() - debut)
    except TropDeMotifs:
        print("  %-52s ARRÊT : trop d'itemsets, version ignorée" % nom.replace("\n", " "))
        return None

    regles = generer_regles(frequents, D.N, D.bits, confiance_min=CONFIANCE_MIN, lift_min=LIFT_MIN,
                            conclusion_max=CONCLUSION_MAX, separateur=SEPARATEUR)
    lignes = []
    for r in regles:
        elements = r["si"].split(SEPARATEUR) + r["alors"].split(SEPARATEUR)
        lignes.append((round(r["support"] * D.N), any(i in D.rares for i in elements), r))
    fiables = [(rare, r) for s, rare, r in lignes if s >= MIN_FIABLE]

    couverts = set()
    for g in frequents:
        couverts.update(g)

    confs = [r["confiance"] for _, r in fiables]
    lifts = [r["lift"] for _, r in fiables]
    n = len(regles)
    m = {"nom": nom, "famille": famille, "seuils": seuils, "mis": mis,
         "itemsets": len(frequents), "candidats": candidats, "noeuds": noeuds, "temps": statistics.median(temps),
         "regles": n, "fiables": len(fiables), "bruit": n - len(fiables),
         "pct_bruit": 100.0 * (n - len(fiables)) / n if n else 0.0,
         "pct_fiables": 100.0 * len(fiables) / n if n else None,
         "pct_rares": 100.0 * sum(1 for _, rare, _ in lignes if rare) / n if n else 0.0,
         "rares_couverts": len(couverts & D.rares), "items_couverts": len(couverts),
         "pct_rares_gardes": 100.0 * len(couverts & D.rares) / max(1, len(D.rares)),
         "conf_fiables": 100 * statistics.mean(confs) if confs else None,
         "lift_fiables": statistics.median(lifts) if lifts else None,
         "cles": {(r["si"], r["alors"]) for _, r in fiables},
         "cles_rares": {(r["si"], r["alors"]) for rare, r in fiables if rare}}
    print("  %-52s %7d itemsets %7d règles (%5.1f %% bruit) %2d/%d rares  %.2f s"
          % (nom.replace("\n", " "), m["itemsets"], n, m["pct_bruit"], m["rares_couverts"], len(D.rares), m["temps"]))
    return m


# 4. Rapport : garde le texte et les graphes, puis les enregistre.
class Rapport:
    def __init__(self, dossier):
        self.dossier, self.lignes, self.figures = dossier, [], []
        os.makedirs(dossier, exist_ok=True)

    def ecrire(self, txt=""):
        print(txt)
        self.lignes.append(txt)

    def sauver_figure(self, fig, nom):
        fig.savefig(os.path.join(self.dossier, nom), dpi=140, facecolor=fig.get_facecolor())
        self.figures.append(fig)

    def enregistrer(self):
        with PdfPages(os.path.join(self.dossier, "graphes_comparaison_approches.pdf")) as pdf:
            for fig in self.figures:
                pdf.savefig(fig)
        with open(os.path.join(self.dossier, "rapport_comparaison_approches.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lignes))
        print("\nÉcrit dans %s : %d graphes PNG, graphes_comparaison_approches.pdf, rapport_comparaison_approches.txt, resultats.csv"
              % (self.dossier, len(self.figures)))


# 5. Outils de dessin (simples : barres horizontales, un message par graphe).
def style(ax, grille="x"):
    ax.set_facecolor(FOND)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.grid(axis=grille, alpha=.3)
    ax.set_axisbelow(True)


# Titre (ce que montre le graphe), sous-titre (le résultat en une phrase) et "comment lire" en bas.
def en_tete(fig, titre, sous_titre, comment_lire):
    fig.text(0.015, 0.975, titre, fontsize=15, fontweight="bold", ha="left", va="top", color=ENCRE)
    fig.text(0.015, 0.915, textwrap.fill(sous_titre, 115), fontsize=10.5, ha="left", va="top", color=ENCRE_2)
    fig.text(0.015, 0.015, textwrap.fill("Comment lire : " + comment_lire, 140), fontsize=9, ha="left", va="bottom", color=ENCRE_2)
    fig.tight_layout(rect=(0, 0.10, 1, 0.83))


# Valeur d'une version : moyenne de ses tirages (un seul tirage pour la plupart des versions).
def val(g, cle):
    vs = [r[cle] for r in g["runs"] if r.get(cle) is not None]
    return statistics.mean(vs) if vs else None


# Barres horizontales : une barre par version, la valeur écrite au bout.
def barres_h(ax, groupes, valeur, texte, xmax=None, pourcentage=False, etiquettes=True, ligne=None):
    style(ax)
    ax.spines["left"].set_visible(False)
    vals = [valeur(g) for g in groupes]
    connues = [v for v in vals if v is not None]
    if pourcentage:
        xmax = 125
    elif xmax is None:
        xmax = max(connues + [1e-9]) * 1.35
    for i, (g, v) in enumerate(zip(groupes, vals)):
        if v is None:
            ax.text(0.01 * xmax, i, "aucune règle fiable", va="center", fontsize=9, color=ENCRE_2)
            continue
        ax.barh(i, v, height=0.55, color=g["couleur"], zorder=2)
        ax.text(v + 0.012 * xmax, i, texte(g, v), va="center", fontsize=10, fontweight="bold", color=ENCRE)
    ax.set_yticks(range(len(groupes)))
    ax.set_yticklabels([g["nom"] for g in groupes] if etiquettes else [], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="y", length=0)
    if pourcentage:
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=9)
    if ligne is not None:
        ax.axvline(ligne, color=ENCRE_2, ls="--", lw=1)


# Seuil minimum médian (en tickets) des items d'une catégorie, pour une version.
def seuil_median(run, D, categorie):
    s = [run["seuils"][p] for p in D.nb if D.cat[p] == categorie]
    return statistics.median(s) if s else None


# 6. Les 8 graphiques.
def graphes(rapport, groupes, notre, reference, D, balayage_beta, balayage_ls):
    nom_notre = "Notre approche"
    nb_rares = len(D.rares)

    # 1. Seuils : à partir de combien de tickets un item est-il considéré comme fréquent ?
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), facecolor=FOND)
    for c, ax in enumerate(axes):
        barres_h(ax, groupes, lambda g, c=c: statistics.mean(seuil_median(r, D, c) for r in g["runs"]),
                 lambda g, v: mf(v), etiquettes=(c == 0))
        ax.set_title("Items de fréquence %s (%d items)" % (NOMS_CAT[c], D.nb_cat[c]), loc="left", fontsize=10.5, color=ENCRE)
        ax.set_xlabel("seuil minimum médian (en tickets)", fontsize=9)
    en_tete(fig, "1. À partir de combien de tickets un item compte-t-il comme « fréquent » ?",
            "Plus le seuil est bas, plus on garde d'items rares. Notre approche adapte le seuil au type d'item : bas pour les rares, "
            "plus haut pour les très fréquents.",
            "chaque barre est le seuil médian de la version, en nombre de tickets. Trois graphes : items rares (faible), "
            "moyens et très fréquents (forte). Vert = notre approche.")
    rapport.sauver_figure(fig, "1_seuils_par_type_d_item.png")

    # 2. Items rares conservés.
    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=FOND)
    barres_h(ax, groupes, lambda g: val(g, "pct_rares_gardes"), lambda g, v: mf(v) + " %", pourcentage=True)
    ax.set_xlabel("part des items rares présents dans au moins un itemset fréquent", fontsize=9.5)
    en_tete(fig, "2. Combien d'items rares sont conservés ?",
            "Notre approche garde %s %% des %d items rares (vendus dans moins de %s %% des tickets)."
            % (mf(notre["pct_rares_gardes"]), nb_rares, mf(100 * RARE)),
            "plus la barre est longue, moins la version perd d'items rares. 100 % = aucun item rare perdu.")
    rapport.sauver_figure(fig, "2_items_rares_conserves.png")

    # 3. Règles fiables retrouvées.
    if reference:
        fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=FOND)
        barres_h(ax, groupes, lambda g: val(g, "rappel"), lambda g, v: mf(v) + " %", pourcentage=True)
        ax.set_xlabel("part des règles fiables de la référence retrouvées par la version", fontsize=9.5)
        en_tete(fig, "3. Combien de bonnes règles retrouve-t-on ?",
                "Notre approche retrouve %s %% des règles fiables (au moins %d tickets) que donnerait un seuil de 1 ticket."
                % (mf(notre["rappel"]), MIN_FIABLE),
                "la référence est la recherche la plus complète possible (seuil de 1 ticket). 100 % = aucune bonne règle manquée.")
        rapport.sauver_figure(fig, "3_regles_fiables_retrouvees.png")

    # 4. Fiables parmi toutes les règles.
    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=FOND)
    barres_h(ax, groupes, lambda g: val(g, "pct_fiables"),
             lambda g, v: "%s %%  (%s règles)" % (mf(v), mf(val(g, "regles"))), pourcentage=True)
    ax.set_xlim(0, 150)
    ax.set_xlabel("part des règles produites qui reposent sur au moins %d tickets" % MIN_FIABLE, fontsize=9.5)
    en_tete(fig, "4. Parmi les règles produites, combien sont fiables ?",
            "Une règle est fiable si elle repose sur au moins %d tickets. Les autres sont du bruit (coïncidences)." % MIN_FIABLE,
            "plus la barre est longue, moins la version produit de règles inutiles. Entre parenthèses : le nombre total de règles.")
    rapport.sauver_figure(fig, "4_regles_fiables_contre_bruit.png")

    # 5. Qualité : confiance et lift.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=FOND)
    barres_h(a1, groupes, lambda g: val(g, "conf_fiables"), lambda g, v: mf(v) + " %", pourcentage=True)
    a1.set_title("Confiance moyenne", loc="left", fontsize=10.5, color=ENCRE)
    a1.set_xlabel("« si le client achète A, il achète B dans … % des cas »", fontsize=9)
    barres_h(a2, groupes, lambda g: val(g, "lift_fiables"), lambda g, v: mf(v, 1), etiquettes=False, ligne=1)
    a2.set_title("Lift médian", loc="left", fontsize=10.5, color=ENCRE)
    a2.set_xlabel("lift = nombre de fois où A et B sont achetés ensemble plus que par hasard", fontsize=9)
    en_tete(fig, "5. Les règles trouvées sont-elles de bonne qualité ?",
            "Mesure faite sur les règles fiables uniquement. Notre approche : confiance %s %%, lift %s."
            % (mf(notre["conf_fiables"]) if notre["conf_fiables"] is not None else "-",
               mf(notre["lift_fiables"], 1) if notre["lift_fiables"] is not None else "-"),
            "plus c'est long, mieux c'est. Pour le lift, le trait pointillé (1) signifie « aucun lien » : "
            "au-dessus, les produits sont vraiment achetés ensemble.")
    rapport.sauver_figure(fig, "5_qualite_des_regles.png")

    # 6. Temps.
    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=FOND)
    barres_h(ax, groupes, lambda g: val(g, "temps"), lambda g, v: mf(v, 2) + " s")
    ax.set_xlabel("temps de CFP-Growth en secondes (médiane de %d exécutions)" % REPETITIONS, fontsize=9.5)
    en_tete(fig, "6. Combien de temps le calcul prend-il ?",
            "Notre approche : %s s." % mf(notre["temps"], 2),
            "plus la barre est courte, plus la version est rapide. Même moteur CFP-Growth pour toutes les versions.")
    rapport.sauver_figure(fig, "6_temps_de_calcul.png")

    # 7. Effet des réglages β et LS de la formule originale.
    def reglage(ax, xs, runs, xlabel, titre):
        style(ax, "both")
        ax.set_xscale("log")
        ax.plot(xs, [r["pct_rares_gardes"] for r in runs], "o-", color=ORANGE, lw=2.2, ms=6, label="items rares conservés (formule originale)")
        ax.axhline(notre["pct_rares_gardes"], color=VERT, ls="--", lw=2, label="items rares conservés (notre approche)")
        if reference:
            ax.plot(xs, [r["rappel"] for r in runs], "s-", color=BLEU_FONCE, lw=2.2, ms=6, label="règles fiables retrouvées (formule originale)")
            ax.axhline(notre["rappel"], color=VERT, ls="-.", lw=2, label="règles fiables retrouvées (notre approche)")
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_title(titre, loc="left", fontsize=10.5, color=ENCRE)
        ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)

    bs = [b for b, r in balayage_beta if r]
    rb = [r for b, r in balayage_beta if r]
    ls_x = [ls for ls, r in balayage_ls if r]
    rl = [r for ls, r in balayage_ls if r]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.4), facecolor=FOND, sharey=True)
    reglage(a1, bs, rb, "valeur de β choisie à la main (échelle log)", "Si on change β (LS = %d ticket%s)" % (LS_FIXE, "s" if LS_FIXE > 1 else ""))
    reglage(a2, ls_x, rl, "valeur de LS choisie à la main, en tickets (échelle log)", "Si on change LS (β = %s)" % fr(BETA_FIXE))
    en_tete(fig, "7. La formule originale dépend des réglages choisis à la main",
            "Les courbes bougent selon β et LS ; notre approche (traits verts) n'a aucun réglage à choisir.",
            "chaque point est un essai de la formule originale avec une valeur de β (à gauche) ou de LS (à droite). "
            "Plus la courbe est haute, mieux c'est. Les traits verts horizontaux montrent notre approche.")
    rapport.sauver_figure(fig, "7_effet_des_reglages.png")

    # 8. Tableau de synthèse.
    colonnes = [("Items rares\nconservés", "pct_rares_gardes", True, lambda v: mf(v) + " %"),
                ("Règles fiables\nretrouvées", "rappel", True, lambda v: mf(v) + " %"),
                ("Part de règles\nfiables", "pct_fiables", True, lambda v: mf(v) + " %"),
                ("Confiance\nmoyenne", "conf_fiables", True, lambda v: mf(v) + " %"),
                ("Temps", "temps", False, lambda v: mf(v, 2) + " s")]
    if not reference:
        colonnes = [c for c in colonnes if c[1] != "rappel"]
    table = [[val(g, c[1]) for c in colonnes] for g in groupes]
    meilleur = []
    for j, c in enumerate(colonnes):
        v = [ligne[j] for ligne in table if ligne[j] is not None]
        meilleur.append((max(v) if c[2] else min(v)) if v else None)
    textes = [[colonnes[j][3](x) if x is not None else "-" for j, x in enumerate(ligne)] for ligne in table]
    couleurs = [["#cfeadb" if x is not None and meilleur[j] is not None and abs(x - meilleur[j]) < 1e-9 else FOND
                 for j, x in enumerate(ligne)] for ligne in table]
    fig, ax = plt.subplots(figsize=(13, 5.6), facecolor=FOND)
    ax.axis("off")
    tab = ax.table(cellText=textes, cellColours=couleurs, rowLabels=[g["nom"] for g in groupes],
                   colLabels=[c[0] for c in colonnes], loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10.5)
    tab.scale(1, 2.0)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if i == 0 or j == -1:
            cell.set_text_props(fontweight="bold", color=ENCRE)
        if j == -1:
            cell.set_text_props(ha="right")
    en_tete(fig, "8. Synthèse : toutes les versions côte à côte",
            "Fond vert = meilleure valeur de la colonne.",
            "chaque ligne est une version, chaque colonne une mesure expliquée dans les graphes 2 à 6. "
            "Pour les versions aléatoires, c'est la moyenne de %d tirages." % len(GRAINES))
    rapport.sauver_figure(fig, "8_synthese.png")


# 7. Programme principal.
def main():
    rapport = Rapport(DOSSIER_SORTIE)
    if not os.path.exists(FICHIER):
        print("Fichier introuvable :", FICHIER)
        return

    # a) Données.
    tickets = lire_tickets(FICHIER)
    N = len(tickets)
    nb = compter_supports(tickets)
    _, cat = categories_quartiles(nb)
    D = SimpleNamespace(N=N, nb=nb, cat=cat, ensembles=[set(t) for t in tickets], bits=preparer_bitsets(tickets),
                        rares={p for p in nb if nb[p] < RARE * N}, nb_cat=[sum(1 for p in nb if cat[p] == c) for c in range(3)])
    m = calculer_mis_categories(tickets)
    rapport.ecrire("DONNÉES : %d tickets, %d items (rares, support < %g %% : %d) | items par catégorie : faible %d, moyenne %d, forte %d"
                   % (N, len(nb), 100 * RARE, len(D.rares), *D.nb_cat))
    rapport.ecrire("RÈGLES : confiance >= %g %%, lift > %g | itemsets de taille <= %s | règle fiable = au moins %d tickets"
                   % (100 * CONFIANCE_MIN, LIFT_MIN, TAILLE_MAX, MIN_FIABLE))
    rapport.ecrire("NOTRE APPROCHE : bêtas (faible / moyen / fort) = %s | seuils de %d à %d tickets"
                   % (" / ".join("%.5f" % b for b in m["betas"]), min(m["seuils"].values()), max(m["seuils"].values())))
    mis_bas, mis_haut = min(m["mis"].values()), max(m["mis"].values())
    plage_notre = "%s–%s %%" % (fr(round(100 * mis_bas, 2)), fr(round(100 * mis_haut, 2)))
    plage_large = "%s–%s %%" % (fr(round(100 * MIS_MIN, 1)), fr(round(100 * MIS_MAX)))

    # b) Lancer toutes les versions.
    rapport.ecrire("\n--- Exécution (chaque ligne = une version) ---")
    cache = {}

    def lancer_originale(beta, ls):
        if (beta, ls) not in cache:
            nom = "Originale β=%s, LS=%d ticket%s" % (fr(beta), ls, "s" if ls > 1 else "")
            cache[(beta, ls)] = evaluer(nom, "originale", *mis_originale(nb, N, beta, ls), D)
        return cache[(beta, ls)]

    notre = evaluer("Notre approche", "notre", *mis_notre(m), D)
    reference = evaluer("Référence : seuil unique 1 ticket", "reference", *mis_unique(nb, N, 1), D) if AVEC_REFERENCE else None
    balayage_beta = [(b, lancer_originale(b, LS_FIXE)) for b in BETAS_BALAYAGE]
    balayage_ls = [(ls, lancer_originale(BETA_FIXE, ls)) for ls in LS_BALAYAGE]
    originales = [lancer_originale(b, ls) for b, ls in ORIGINALES]

    def tirages(famille, bas, haut):
        return [r for r in (evaluer("Aléatoire %s, graine %d" % (famille, g), "aleatoire " + famille,
                                    *mis_aleatoire(nb, N, bas, haut, g), D) for g in GRAINES) if r]

    alea_large = tirages("large", MIS_MIN, MIS_MAX)
    alea_meme = tirages("même plage", mis_bas, mis_haut)

    # c) Recouvrement avec la référence : quelles règles fiables sont retrouvées ?
    toutes = [r for r in [notre, reference] + [x[1] for x in balayage_beta] + [x[1] for x in balayage_ls]
              + originales + alea_large + alea_meme if r]
    toutes = list({id(r): r for r in toutes}.values())
    for r in toutes:
        if reference:
            r["rappel"] = 100.0 * len(r["cles"] & reference["cles"]) / max(1, len(reference["cles"]))
            r["rappel_rares"] = 100.0 * len(r["cles_rares"] & reference["cles_rares"]) / max(1, len(reference["cles_rares"]))
        else:
            r["rappel"] = r["rappel_rares"] = None

    # d) Versions affichées dans les graphes : un nom lisible et une couleur par famille.
    groupes = [{"nom": "Notre approche (automatique)", "couleur": VERT, "runs": [notre]}]
    for i, (b, ls) in enumerate(ORIGINALES):
        groupes.append({"nom": "Originale : β = %s, LS = %d" % (fr(b), ls), "couleur": BLEU, "runs": [originales[i]]})
    groupes.append({"nom": "Aléatoire, plage large %s (moy. de %d)" % (plage_large, len(GRAINES)), "couleur": GRIS, "runs": alea_large})
    groupes.append({"nom": "Aléatoire, même plage que nous %s (moy. de %d)" % (plage_notre, len(GRAINES)), "couleur": ORANGE, "runs": alea_meme})
    groupes = [g for g in groupes if g["runs"] and all(g["runs"])]

    # e) Tableau récapitulatif (texte).
    rapport.ecrire("\n=== TABLEAU RÉCAPITULATIF (moyenne des tirages) ===")
    colonnes = [("itemsets", "Itemsets", "%.0f"), ("candidats", "Candidats", "%.0f"), ("regles", "Règles", "%.0f"),
                ("fiables", "Fiables", "%.0f"), ("pct_bruit", "% bruit", "%.1f"), ("rares_couverts", "Rares gardés", "%.0f"),
                ("rappel", "Rappel fiables %", "%.1f"), ("temps", "Temps (s)", "%.2f")]
    rapport.ecrire("%-50s " % "Version" + " ".join("%16s" % c[1] for c in colonnes))
    for g in groupes:
        cellules = []
        for cle, _, fmt in colonnes:
            v = [r[cle] for r in g["runs"] if r[cle] is not None]
            cellules.append("%16s" % (fmt % statistics.mean(v) if v else "-"))
        rapport.ecrire("%-50s " % g["nom"] + " ".join(cellules))

    # f) Graphiques.
    graphes(rapport, groupes, notre, reference, D, balayage_beta, balayage_ls)

    # g) Résultats bruts en CSV.
    with open(os.path.join(DOSSIER_SORTIE, "resultats.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        champs = ["itemsets", "candidats", "noeuds", "regles", "fiables", "bruit", "pct_bruit", "pct_rares",
                  "rares_couverts", "items_couverts", "rappel", "rappel_rares", "conf_fiables", "lift_fiables", "temps"]
        w.writerow(["version", "famille"] + champs)
        for r in toutes:
            w.writerow([r["nom"].replace("\n", " "), r["famille"]] + [r[c] for c in champs])

    rapport.enregistrer()


if __name__ == "__main__":
    main()