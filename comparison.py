# =====================================================================
#  COMPARISON.PY  -  compare Apriori et CFP-Growth (MIS par quartiles)
#
#  POUR LANCER :   python comparison.py
#  POUR CHANGER LES RÉGLAGES : modifier les valeurs juste en dessous (RÉGLAGES), puis relancer.
#
#  Sommaire du fichier (dans l'ordre de lecture) :
#     1. Load the dataset                -> charger_donnees
#     2. Compare Apriori and CFP-Growth  -> apriori_original, comparer_meme_seuil, tester_taille_base,
#                                           comparer_apriori_quartiles, verifier_et_mesurer
#     3. Test different LS values        -> tester_valeurs_ls
#     4. Test different beta values      -> tester_valeurs_beta
#     5. Measure time and memory         -> chrono, pic_memoire
#     6. Generate graphs and reports     -> Rapport, style, etiquettes
#     7. Programme principal             -> main  (l'ordre des étapes est là)
#
#  Un "beta" est ici un triplet (beta_faible, beta_moyen, beta_fort) :
#  CFP-Growth calcule le MIS de chaque produit avec le beta de sa catégorie (quartiles).
# =====================================================================
import math
import os
import statistics
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")                     # dessiner dans des fichiers, sans ouvrir de fenêtre
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cfp_growth import cfp_growth, cfp_growth_seuils, compter_supports
from main import lire_tickets             # même lecture de données que main.py

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
#  RÉGLAGES  (les seules lignes à modifier)
# =====================================================================
FICHIER = os.path.join(DOSSIER, "Data", "diabetic_data_apriori.csv")

# Le diabète est très dense (environ 22 items par patient) : il faut des LS assez grands.
LS_LIST = [0.5, 0.4, 0.3, 0.25, 0.2]         # valeurs de LS testées (du plus grand au plus petit)
LS_FIXE = 0.2                                # LS utilisé quand il ne varie pas (doit être dans LS_LIST)
TAILLES = [5000, 10000, 20000, 40000]        # tailles de base testées (la base complète s'ajoute)
RARE = 0.3                                   # un item est "rare" si son support < 30 %

BETAS = (0.2, 0.5, 0.8)                      # beta utilisé quand il ne varie pas (faible, moyen, fort)
BETAS_LISTE = [(0.1, 0.3, 0.5),              # les beta testés dans "Test different beta values"
               (0.2, 0.5, 0.8),
               (0.3, 0.5, 0.7),
               (0.5, 0.7, 0.9)]
REPETITIONS = 3                              # chaque mesure de temps est répétée 3 fois (on garde la médiane)

DOSSIER_SORTIE = os.path.join(DOSSIER, "resultats_comparaison")

FOND = "#fdfcfa"
BLEU, VERT, GRIS, ORANGE = "#2878d6", "#2e9f6e", "#9b9a94", "#e8742f"
LARGEUR_BARRE = 0.38


# =====================================================================
#  1. LOAD THE DATASET
# =====================================================================
def charger_donnees(rapport):
    """Lit le fichier. Renvoie (tickets, produits rares)."""
    tickets = lire_tickets(FICHIER)
    N = len(tickets)
    nb_prod = compter_supports(tickets)
    rares = {p for p in nb_prod if nb_prod[p] < RARE * N}      # produits rares (support faible)
    rapport.ecrire("DONNÉES : %d tickets, %d produits (rares, support < %g %% : %d)"
                   % (N, len(nb_prod), 100 * RARE, len(rares)))
    rapport.ecrire("RÉGLAGES : LS = %s | beta = %s | LS fixe = %g %% | médiane de %d exécutions"
                   % ([round(100 * x, 2) for x in LS_LIST], etiquette_beta(BETAS), 100 * LS_FIXE, REPETITIONS))
    return tickets, rares


# =====================================================================
#  2. COMPARE APRIORI AND CFP-GROWTH
# =====================================================================

# ---------------------------------------------------------------------
#  Apriori original (comme dans le cours, sans aucune amélioration)
#   - un seul seuil MinSup pour tous les produits
#   - recherche par niveau : produits seuls, puis paires, puis trios...
#   - le support de chaque candidat se calcule en relisant tous les tickets
# ---------------------------------------------------------------------
def apriori_original(tickets, minsup):
    """minsup : seuil en fraction (0.4 = 40 % des tickets)."""
    N = len(tickets)
    seuil = math.ceil(round(minsup * N, 9))        # nombre minimum de tickets
    tickets_ens = [set(t) for t in tickets]

    # Étape 1 : les candidats de taille 1 sont tous les produits
    produits = sorted(set().union(*tickets_ens))
    candidats = [(p,) for p in produits]

    frequents = {}                 # {groupe : nombre de tickets} tous niveaux confondus
    candidats_par_niveau = {}      # {k : liste des candidats testés au niveau k}
    frequents_par_niveau = {}      # {k : liste des groupes fréquents du niveau k}
    k = 1

    while len(candidats) > 0:
        candidats_par_niveau[k] = list(candidats)

        # Étapes 2 et 3 : support de chaque candidat (on relit les tickets), on garde >= MinSup
        frequents_k = []
        for groupe in candidats:
            ens = set(groupe)
            compteur = 0
            for ticket in tickets_ens:
                if ens <= ticket:                  # le ticket contient tout le groupe
                    compteur += 1
            if compteur >= seuil:
                frequents[groupe] = compteur
                frequents_k.append(groupe)
        frequents_par_niveau[k] = list(frequents_k)

        # Étape 4 : candidats du niveau suivant
        #   a) jointure : on colle deux groupes qui ont le même début
        #   b) élagage  : tous les sous-groupes du candidat doivent être fréquents
        frequents_k.sort()
        frequents_ensemble = set(frequents_k)
        debuts = {}
        for groupe in frequents_k:
            debuts.setdefault(groupe[:-1], []).append(groupe[-1])

        candidats = []
        for debut, fins in debuts.items():
            for x in range(len(fins)):
                for y in range(x + 1, len(fins)):
                    candidat = debut + (fins[x], fins[y])
                    ok = True
                    for i in range(len(candidat)):
                        sous_groupe = candidat[:i] + candidat[i + 1:]
                        if sous_groupe not in frequents_ensemble:
                            ok = False
                            break
                    if ok:
                        candidats.append(candidat)
        candidats.sort()
        k += 1

    return {"frequents": frequents,
            "candidats": candidats_par_niveau,
            "frequents_par_niveau": frequents_par_niveau}


def seuil_unique(tickets, minsup):
    """Le même seuil (en nombre de tickets) pour tous les produits : c'est ce que fait Apriori."""
    s = math.ceil(round(minsup * len(tickets), 9))
    return {p: s for t in tickets for p in t}


def avec_rare(frequents, rares):
    """Nombre d'itemsets qui contiennent au moins un produit rare."""
    return sum(1 for g in frequents if any(p in rares for p in g))


def etiquette_beta(betas):
    """(0.2, 0.5, 0.8) -> '0.2/0.5/0.8'"""
    return "/".join("%g" % b for b in betas)


def ligne_pour(lignes, ls):
    """Retrouve dans les résultats de la partie 1 la ligne qui correspond à ce LS."""
    for ligne in lignes:
        if ligne["ls"] == ls:
            return ligne


def comparer_meme_seuil(tickets, rapport):
    """PARTIE 1 : Apriori et CFP-Growth avec le MÊME seuil unique (comparaison équitable).
    Ils doivent trouver exactement les mêmes itemsets. Renvoie la liste des résultats."""
    rapport.ecrire("\n=== PARTIE 1 : même seuil unique pour les deux (comparaison équitable) ===")
    rapport.ecrire("%-8s | %-24s | %-26s | mêmes itemsets" % ("seuil", "Apriori (s | candidats)", "CFP-growth (s | motifs)"))
    lignes = []
    for ls in LS_LIST:
        a, ta = chrono(lambda: apriori_original(tickets, ls))
        c, tc = chrono(lambda: cfp_growth_seuils(tickets, seuil_unique(tickets, ls)))
        nca = sum(len(v) for v in a["candidats"].values())
        nce = sum(c["enumeres_par_niveau"].values())
        meme = set(a["frequents"]) == set(c["frequents"])
        lignes.append(dict(ls=ls, ta=ta, tc=tc, nca=nca, nce=nce, a=a, c=c))
        rapport.ecrire("%-8s | %6.2f s | %6d cand. | %6.2f s | %6d motifs | %s"
                       % ("%g %%" % (100 * ls), ta, nca, tc, nce, meme))

    # Graphe 1 : temps selon le seuil
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND); style(ax)
    x = range(len(lignes))
    ax.bar([i - LARGEUR_BARRE / 2 for i in x], [r["ta"] for r in lignes], LARGEUR_BARRE, color=BLEU, label="Apriori original")
    ax.bar([i + LARGEUR_BARRE / 2 for i in x], [r["tc"] for r in lignes], LARGEUR_BARRE, color=VERT, label="CFP-growth")
    ax.set_xticks(list(x)); ax.set_xticklabels(["%g %%" % (100 * r["ls"]) for r in lignes])
    ax.set_xlabel("seuil unique (= LS)"); ax.set_ylabel("temps (s)"); ax.legend(frameon=False)
    ax.set_title("1. Temps selon le seuil (même seuil pour les deux)", loc="left", fontsize=10)
    rapport.sauver_figure(fig, "1_temps_selon_seuil.png")

    # Graphe 2 : candidats par niveau (au seuil LS_FIXE)
    ref = ligne_pour(lignes, LS_FIXE)
    niv = sorted(set(ref["a"]["candidats"]) | set(ref["c"]["enumeres_par_niveau"]))
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND); style(ax)
    ax.bar([i - LARGEUR_BARRE / 2 for i in range(len(niv))], [len(ref["a"]["candidats"].get(k, [])) for k in niv],
           LARGEUR_BARRE, color=BLEU, label="Apriori : candidats testés")
    ax.bar([i + LARGEUR_BARRE / 2 for i in range(len(niv))], [ref["c"]["enumeres_par_niveau"].get(k, 0) for k in niv],
           LARGEUR_BARRE, color=VERT, label="CFP-growth : motifs trouvés (0 candidat généré)")
    ax.set_xticks(range(len(niv))); ax.set_xticklabels(["taille %d" % k for k in niv])
    ax.set_ylabel("nombre"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("2. Candidats par niveau (seuil %g %%)" % (100 * LS_FIXE), loc="left", fontsize=10)
    rapport.sauver_figure(fig, "2_candidats_par_niveau.png")
    return lignes


def tester_taille_base(tickets, rapport):
    """PARTIE 2 : temps selon la taille de la base (LS fixe, même seuil unique pour les deux)."""
    rapport.ecrire("\n=== PARTIE 2 : temps selon la taille de la base (seuil %g %%) ===" % (100 * LS_FIXE))
    ns, ta_l, tc_l = [], [], []
    for n in [t for t in TAILLES if t < len(tickets)] + [len(tickets)]:
        T = tickets[:n]
        _, ta = chrono(lambda: apriori_original(T, LS_FIXE))
        _, tc = chrono(lambda: cfp_growth_seuils(T, seuil_unique(T, LS_FIXE)))
        ns.append(n); ta_l.append(ta); tc_l.append(tc)
        rapport.ecrire("N = %5d : Apriori %.2f s | CFP-growth %.2f s | rapport x%.1f" % (n, ta, tc, ta / max(tc, 1e-9)))

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND); style(ax)
    ax.plot(ns, ta_l, "o-", color=BLEU, label="Apriori original")
    ax.plot(ns, tc_l, "s-", color=VERT, label="CFP-growth")
    ax.set_xlabel("nombre de tickets"); ax.set_ylabel("temps (s)"); ax.legend(frameon=False)
    ax.set_title("3. Temps selon la taille de la base (seuil %g %%)" % (100 * LS_FIXE), loc="left", fontsize=10)
    rapport.sauver_figure(fig, "3_temps_selon_taille.png")


def comparer_apriori_quartiles(tickets, rares, rapport, lignes):
    """PARTIE 5 : Apriori (un seuil pour tous) contre CFP-Growth (un MIS par produit, quartiles)."""
    rapport.ecrire("\n=== PARTIE 5 : Apriori (minsup = LS = %g %%) contre CFP-growth (LS, beta = %s) ==="
                   % (100 * LS_FIXE, etiquette_beta(BETAS)))
    ref = ligne_pour(lignes, LS_FIXE)                  # Apriori déjà calculé en partie 1
    c, tc = chrono(lambda: cfp_growth(tickets, LS_FIXE, BETAS))
    fa, fc = set(ref["a"]["frequents"]), set(c["frequents"])
    rapport.ecrire("Apriori : %d itemsets, %.2f s" % (len(fa), ref["ta"]))
    rapport.ecrire("CFP-growth (beta = %s) : %d itemsets, %.2f s" % (etiquette_beta(BETAS), len(fc), tc))
    rapport.ecrire("en commun : %d | seulement Apriori : %d | seulement CFP-growth : %d"
                   % (len(fa & fc), len(fa - fc), len(fc - fa)))
    rapport.ecrire("itemsets avec un produit rare (support < %g %%) : Apriori %d | CFP-growth %d"
                   % (100 * RARE, avec_rare(fa, rares), avec_rare(fc, rares)))

    # Graphe 8 : itemsets en commun et différents, par taille
    niv = sorted({len(g) for g in fa | fc})
    comm = [sum(1 for g in fa & fc if len(g) == k) for k in niv]
    sa = [sum(1 for g in fa - fc if len(g) == k) for k in niv]
    sm = [sum(1 for g in fc - fa if len(g) == k) for k in niv]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND); style(ax)
    ax.bar(range(len(niv)), comm, color=GRIS, label="en commun")
    ax.bar(range(len(niv)), sa, bottom=comm, color=BLEU, label="seulement Apriori")
    ax.bar(range(len(niv)), sm, bottom=[x + y for x, y in zip(comm, sa)], color=VERT, label="seulement CFP-growth")
    ax.set_xticks(range(len(niv))); ax.set_xticklabels(["taille %d" % k for k in niv])
    ax.set_ylabel("itemsets fréquents"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("8. Itemsets : Apriori (1 seuil) contre CFP-growth (quartiles, β = %s)" % etiquette_beta(BETAS),
                 loc="left", fontsize=9)
    rapport.sauver_figure(fig, "8_itemsets_communs_et_differents.png")


def verifier_et_mesurer(tickets, rapport, lignes):
    """PARTIE 6 : vérification (mêmes itemsets avec le même seuil) + comparaison globale temps / mémoire."""
    ref = ligne_pour(lignes, LS_FIXE)
    a, c = ref["a"]["frequents"], ref["c"]["frequents"]

    # Graphe 9 : vérification
    niv = sorted({len(g) for g in a} | {len(g) for g in c})
    na = [sum(1 for g in a if len(g) == k) for k in niv]
    nc = [sum(1 for g in c if len(g) == k) for k in niv]
    identiques = set(a) == set(c)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=FOND); style(ax)
    b1 = ax.bar([i - LARGEUR_BARRE / 2 for i in range(len(niv))], na, LARGEUR_BARRE, color=BLEU,
                label="Apriori (seuil %g %%)" % (100 * LS_FIXE))
    b2 = ax.bar([i + LARGEUR_BARRE / 2 for i in range(len(niv))], nc, LARGEUR_BARRE, color=VERT, label="CFP-growth (même seuil)")
    etiquettes(ax, b1); etiquettes(ax, b2)
    ax.set_xticks(range(len(niv))); ax.set_xticklabels(["Niveau %d" % k for k in niv])
    ax.set_ylabel("Itemsets fréquents"); ax.legend(frameon=False)
    ax.set_title("9. Vérification : CFP-growth trouve les mêmes itemsets qu'Apriori\n%d tickets. Itemsets identiques : %s (%d itemsets au total)"
                 % (len(tickets), "OUI" if identiques else "NON", len(a)), loc="left", fontsize=10)
    rapport.sauver_figure(fig, "9_verification_cfp.png")

    # Graphe 10 : comparaison globale (temps ET mémoire)
    mem_a = pic_memoire(lambda: apriori_original(tickets, LS_FIXE))
    mem_c = pic_memoire(lambda: cfp_growth_seuils(tickets, seuil_unique(tickets, LS_FIXE)))
    valeurs = {
        "Itemsets fréquents trouvés": (len(a), len(c), "%d"),
        "Candidats testés (Apriori)\nou motifs fréquents (CFP)": (ref["nca"], ref["nce"], "%d"),
        "Passages sur la base\n(un par niveau, ou 2 pour CFP)": (len(ref["a"]["candidats"]), ref["c"]["passages"], "%d"),
        "Temps d'exécution (s)": (ref["ta"], ref["tc"], "%.2f"),
        "Mémoire utilisée, pic (Mo)": (mem_a, mem_c, "%.1f"),
    }
    rapport.ecrire("\n=== COMPARAISON DES DEUX ALGORITHMES (seuil unique %g %% pour les deux) ===" % (100 * LS_FIXE))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), facecolor=FOND)
    for ax, (titre, (va, vc, fmt)) in zip(axes.flat, valeurs.items()):
        style(ax)
        barres = ax.bar(["Apriori", "CFP-\nGrowth"], [va, vc], color=[BLEU, VERT], width=0.55)
        etiquettes(ax, barres, fmt)
        ax.set_ylim(0, max(va, vc, 1e-9) * 1.18)
        ax.set_title(titre, loc="left", fontsize=9, fontweight="bold")
        rapport.ecrire("%-52s Apriori %s | CFP-growth %s" % (titre.replace("\n", " "), fmt % va, fmt % vc))
    ax = axes.flat[5]; ax.axis("off")
    ax.text(0, 0.95, "Comment lire ces graphes", fontweight="bold", va="top", fontsize=10)
    ax.text(0, 0.82, "Apriori et CFP-Growth utilisent le même\nseuil unique : mêmes itemsets.\n\n"
            "CFP-Growth ne génère pas de candidats :\nil énumère les motifs fréquents dans l'arbre.\n\n"
            "Apriori relit la base à chaque niveau.\nCFP-Growth la lit 2 fois.",
            va="top", fontsize=9, color="#444")
    fig.suptitle("10. Comparaison des deux algorithmes (temps et mémoire)", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.text(0.01, 0.925, "%d tickets. Apriori et CFP-Growth : seuil unique = %g %%" % (len(tickets), 100 * LS_FIXE),
             fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    rapport.sauver_figure(fig, "10_comparaison_deux_algorithmes.png")


# =====================================================================
#  3. TEST DIFFERENT LS VALUES
#     Apriori : minsup = LS      |      CFP-Growth : MIS par quartiles (beta fixé)
# =====================================================================
def tester_valeurs_ls(tickets, rares, rapport, lignes):
    rapport.ecrire("\n=== PARTIE 3 : LS varie (beta = %s) ===" % etiquette_beta(BETAS))
    rapport.ecrire("%-7s | %-28s | %-24s | %s" % ("LS", "Apriori (s | motifs | cand.)", "CFP-growth (s | motifs)",
                                                  "motifs avec produit rare (Apriori | CFP)"))
    E = []
    for ls in LS_LIST:
        ligne = ligne_pour(lignes, ls)                 # Apriori (minsup = LS) déjà calculé en partie 1
        a, ta = ligne["a"], ligne["ta"]
        c, tc = chrono(lambda: cfp_growth(tickets, ls, BETAS))
        cand = sum(len(v) for v in a["candidats"].values())
        e = dict(ls=ls, ta=ta, tc=tc, ma=len(a["frequents"]), mc=len(c["frequents"]), cand=cand,
                 ra=avec_rare(a["frequents"], rares), rc=avec_rare(c["frequents"], rares))
        E.append(e)
        rapport.ecrire("%-7s | %6.2f s %7d %9d | %6.2f s %7d       | %d | %d"
                       % ("%g %%" % (100 * ls), ta, e["ma"], cand, tc, e["mc"], e["ra"], e["rc"]))

    xs = [100 * e["ls"] for e in E]
    nom_cfp = "CFP-growth (β = %s)" % etiquette_beta(BETAS)

    # Graphe 4 : temps selon LS
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND); style(ax)
    ax.plot(xs, [e["ta"] for e in E], "o-", color=BLEU, label="Apriori (minsup = LS)")
    ax.plot(xs, [e["tc"] for e in E], "s-", color=VERT, label=nom_cfp)
    ax.set_xlabel("LS (%)"); ax.set_ylabel("temps (s)"); ax.invert_xaxis(); ax.legend(frameon=False)
    ax.set_title("4. Temps selon LS", loc="left", fontsize=10)
    rapport.sauver_figure(fig, "4_temps_selon_LS.png")

    # Graphe 5 : motifs et candidats selon LS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=FOND); style(ax1); style(ax2)
    ax1.plot(xs, [e["ma"] for e in E], "o-", color=BLEU, label="Apriori")
    ax1.plot(xs, [e["mc"] for e in E], "s-", color=VERT, label=nom_cfp)
    ax1.set_xlabel("LS (%)"); ax1.set_ylabel("motifs fréquents"); ax1.invert_xaxis(); ax1.legend(frameon=False)
    ax1.set_title("(a) Motifs fréquents", loc="left", fontsize=10)
    ax2.plot(xs, [e["cand"] for e in E], "o-", color=BLEU, label="Apriori")
    ax2.plot(xs, [0] * len(E), "s-", color=VERT, label="CFP-growth (aucun candidat)")
    ax2.set_xlabel("LS (%)"); ax2.set_ylabel("candidats générés (cumulés)"); ax2.invert_xaxis(); ax2.legend(frameon=False)
    ax2.set_title("(b) Candidats générés", loc="left", fontsize=10)
    fig.suptitle("5. Motifs et candidats selon LS", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    rapport.sauver_figure(fig, "5_motifs_et_candidats_selon_LS.png")


# =====================================================================
#  4. TEST DIFFERENT BETA VALUES
#     LS fixe. Chaque "beta" testé est un triplet (faible / moyen / fort).
# =====================================================================
def tester_valeurs_beta(tickets, rares, rapport, lignes):
    rapport.ecrire("\n=== PARTIE 4 : beta varie (LS = %g %%, Apriori : minsup = LS) ===" % (100 * LS_FIXE))
    ref = ligne_pour(lignes, LS_FIXE)                  # Apriori déjà calculé en partie 1
    ta_fix, nb_apriori = ref["ta"], len(ref["a"]["frequents"])
    rapport.ecrire("Apriori (ne dépend pas de beta) : %.2f s, %d itemsets" % (ta_fix, nb_apriori))
    rapport.ecrire("%-14s | %-9s | %-9s | %-22s | %s" % ("beta (f/m/F)", "CFP (s)", "itemsets", "motifs MIN-fréquents", "avec produit rare"))

    E = []
    for betas in BETAS_LISTE:
        c, tc = chrono(lambda: cfp_growth(tickets, LS_FIXE, betas))
        e = dict(betas=betas, tc=tc, mc=len(c["frequents"]), enum=sum(c["enumeres_par_niveau"].values()),
                 rc=avec_rare(c["frequents"], rares), occ=c["occurrences"], noeuds=c["noeuds_arbre"])
        E.append(e)
        rapport.ecrire("%-14s | %6.2f s | %-9d | %-22d | %d" % (etiquette_beta(betas), tc, e["mc"], e["enum"], e["rc"]))

    noms = [etiquette_beta(e["betas"]) for e in E]
    x = list(range(len(E)))

    # Graphe 6 : influence de beta
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=FOND); style(ax1); style(ax2)
    ax1.plot(x, [e["tc"] for e in E], "s-", color=VERT, label="CFP-growth")
    ax1.axhline(ta_fix, color=BLEU, ls="--", label="Apriori (indépendant de β)")
    ax1.set_xlabel("β (faible / moyen / fort)"); ax1.set_ylabel("temps (s)"); ax1.legend(frameon=False)
    ax1.set_xticks(x); ax1.set_xticklabels(noms, rotation=20, fontsize=8)
    ax1.set_title("(a) Temps selon β", loc="left", fontsize=10)
    ax2.plot(x, [e["mc"] for e in E], "s-", color=VERT, label="CFP-growth : itemsets retournés")
    ax2.plot(x, [e["enum"] for e in E], "d--", color=ORANGE, label="CFP-growth : motifs MIN-fréquents explorés")
    ax2.axhline(nb_apriori, color=BLEU, ls="--", label="Apriori (minsup = LS)")
    ax2.set_xlabel("β (faible / moyen / fort)"); ax2.set_ylabel("nombre d'itemsets"); ax2.legend(frameon=False, fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(noms, rotation=20, fontsize=8)
    ax2.set_title("(b) Itemsets selon β", loc="left", fontsize=10)
    fig.suptitle("6. Influence de β (LS = %g %%)" % (100 * LS_FIXE), x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    rapport.sauver_figure(fig, "6_influence_beta.png")

    # Graphe 7 : compression du MIS-tree selon beta
    rapport.ecrire("\nCompression du MIS-tree selon beta (LS = %g %%) :" % (100 * LS_FIXE))
    for e in E:
        rapport.ecrire("   beta %s : %d produits dans les tickets -> %d nœuds (÷ %.1f)"
                       % (etiquette_beta(e["betas"]), e["occ"], e["noeuds"], e["occ"] / max(e["noeuds"], 1)))
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=FOND); style(ax)
    b1 = ax.bar([i - LARGEUR_BARRE / 2 for i in x], [e["occ"] for e in E], LARGEUR_BARRE, color=GRIS, label="Produits dans les tickets")
    b2 = ax.bar([i + LARGEUR_BARRE / 2 for i in x], [e["noeuds"] for e in E], LARGEUR_BARRE, color=VERT, label="Nœuds du MIS-tree")
    etiquettes(ax, b1); etiquettes(ax, b2)
    haut = max(max(e["occ"] for e in E), 1)
    for i, e in enumerate(E):
        ax.text(i, haut * 1.08, "÷ %.1f" % (e["occ"] / max(e["noeuds"], 1)), ha="center", color=VERT, fontweight="bold")
    ax.set_ylim(0, haut * 1.2)
    ax.set_xticks(x); ax.set_xticklabels(["β = %s" % n for n in noms], fontsize=8)
    ax.set_ylabel("Nombre"); ax.legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.1))
    ax.set_title("7. Le MIS-tree compresse la base\n(produits à stocker dans les tickets, après filtrage par MIN, contre nœuds de l'arbre)",
                 loc="left", fontsize=10)
    rapport.sauver_figure(fig, "7_compression_arbre.png")


# =====================================================================
#  5. MEASURE TIME AND MEMORY
# =====================================================================
def chrono(f):
    """Exécute f plusieurs fois (REPETITIONS), renvoie (dernier résultat, temps médian en secondes)."""
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


# =====================================================================
#  6. GENERATE GRAPHS AND REPORTS
# =====================================================================
class Rapport:
    """Garde le texte et les graphes, puis les enregistre : PNG, un PDF, un fichier texte."""

    def __init__(self, dossier):
        self.dossier = dossier
        self.lignes = []                   # texte du rapport
        self.figures = []                  # graphes, dans l'ordre
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


def style(ax):
    """Style commun à tous les graphes."""
    ax.set_facecolor(FOND)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)


def etiquettes(ax, barres, fmt="%d"):
    """Écrit la valeur au-dessus de chaque barre."""
    for b in barres:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt % b.get_height(),
                ha="center", va="bottom", fontsize=9)


# =====================================================================
#  7. PROGRAMME PRINCIPAL : l'ordre des étapes
# =====================================================================
def main():
    if LS_FIXE not in LS_LIST:
        raise SystemExit("Erreur dans les RÉGLAGES : LS_FIXE (%g) doit être l'une des valeurs de LS_LIST." % LS_FIXE)

    rapport = Rapport(DOSSIER_SORTIE)

    tickets, rares = charger_donnees(rapport)                         # Load the dataset
    lignes = comparer_meme_seuil(tickets, rapport)                    # Compare (même seuil)
    tester_taille_base(tickets, rapport)                              # Compare (taille de la base)
    tester_valeurs_ls(tickets, rares, rapport, lignes)                # Test different LS values
    tester_valeurs_beta(tickets, rares, rapport, lignes)              # Test different beta values
    comparer_apriori_quartiles(tickets, rares, rapport, lignes)       # Compare (Apriori / quartiles)
    verifier_et_mesurer(tickets, rapport, lignes)                     # Measure time and memory
    rapport.enregistrer()                                             # Graphs and reports


if __name__ == "__main__":
    main()