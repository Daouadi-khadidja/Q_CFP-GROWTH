# Compare Apriori et CFP-Growth (MIS automatique, beta par formule) : temps, mémoire, itemsets, graphes (python comparison.py).
import math
import os
import statistics
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cfp_growth import calculer_ls_auto, cfp_growth, cfp_growth_seuils, compter_supports, quantile_interpolation
from main import lire_tickets

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# RÉGLAGES : les seules valeurs à modifier avant de relancer.
FICHIER = os.path.join(DOSSIER, "Data", "Lim_solar_data_net.csv")

LS_LIST = [0.5, 0.4, 0.3, 0.2, 0.1]          # valeurs de LS testées (base dense : LS assez grands)
LS_FIXE = None                               # None = LS automatique max(1/racine(N), Q3/N), ajouté à LS_LIST au chargement
NIVEAUX_LS = [0.5, 0.75, 0.9]                # niveaux testés pour le quantile de LS (0.75 = Q3, choisi)
TAILLES = [1000, 2500, 5000, 7500]           # tailles de base testées (la base complète s'ajoute)
RARE = 0.3                                   # un item est rare si son support < 30 %

REPETITIONS = 3                              # chaque temps est la médiane de 3 exécutions

DOSSIER_SORTIE = os.path.join(DOSSIER, "resultats_comparaison")

FOND = "#fdfcfa"
BLEU, VERT, GRIS, ORANGE = "#2878d6", "#2e9f6e", "#9b9a94", "#e8742f"
LARGEUR_BARRE = 0.38


# 1. Charge les tickets et repère les produits rares.
def charger_donnees(rapport):
    global LS_FIXE, LS_LIST
    tickets = lire_tickets(FICHIER)
    N = len(tickets)
    nb_prod = compter_supports(tickets)
    if LS_FIXE is None:
        LS_FIXE = calculer_ls_auto(nb_prod, N)
    LS_LIST = sorted(set(LS_LIST + [LS_FIXE]), reverse=True)
    rares = {p for p in nb_prod if nb_prod[p] < RARE * N}
    rapport.ecrire("DONNÉES : %d tickets, %d produits (rares, support < %g %% : %d)"
                   % (N, len(nb_prod), 100 * RARE, len(rares)))
    rapport.ecrire("RÉGLAGES : LS = %s | beta automatique (formule) | LS automatique = %.2f %% | médiane de %d exécutions"
                   % ([round(100 * x, 2) for x in LS_LIST], 100 * LS_FIXE, REPETITIONS))
    return tickets, rares


# 2. Comparaison Apriori / CFP-Growth.
def apriori_original(tickets, minsup):
    """Apriori du cours : un seul seuil, recherche par niveau, supports recalculés en relisant les tickets."""
    N = len(tickets)
    seuil = math.ceil(round(minsup * N, 9))
    tickets_ens = [set(t) for t in tickets]

    # Étape 1 : les candidats de taille 1 sont tous les produits.
    produits = sorted(set().union(*tickets_ens))
    candidats = [(p,) for p in produits]

    frequents = {}
    candidats_par_niveau = {}
    frequents_par_niveau = {}
    k = 1

    while len(candidats) > 0:
        candidats_par_niveau[k] = list(candidats)

        # Étapes 2 et 3 : compter le support de chaque candidat et garder ceux >= MinSup.
        frequents_k = []
        for groupe in candidats:
            ens = set(groupe)
            compteur = 0
            for ticket in tickets_ens:
                if ens <= ticket:
                    compteur += 1
            if compteur >= seuil:
                frequents[groupe] = compteur
                frequents_k.append(groupe)
        frequents_par_niveau[k] = list(frequents_k)

        # Étape 4 : candidats du niveau suivant par jointure puis élagage.
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
    """Même seuil (en nombre de tickets) pour tous les produits, comme Apriori."""
    s = math.ceil(round(minsup * len(tickets), 9))
    return {p: s for t in tickets for p in t}


def avec_rare(frequents, rares):
    """Nombre d'itemsets qui contiennent au moins un produit rare."""
    return sum(1 for g in frequents if any(p in rares for p in g))


def etiquette_beta(betas):
    """(0.2, 0.5, 0.8) devient '0.2/0.5/0.8'."""
    return "/".join(b if isinstance(b, str) else "%g" % b for b in betas)


def ligne_pour(lignes, ls):
    """Retrouve dans les résultats de la partie 1 la ligne de ce LS."""
    for ligne in lignes:
        if ligne["ls"] == ls:
            return ligne


def comparer_meme_seuil(tickets, rapport):
    """PARTIE 1 : Apriori et CFP-Growth avec le même seuil unique, ils doivent trouver les mêmes itemsets."""
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

    # Graphe 1 : temps selon le seuil.
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND); style(ax)
    x = range(len(lignes))
    ax.bar([i - LARGEUR_BARRE / 2 for i in x], [r["ta"] for r in lignes], LARGEUR_BARRE, color=BLEU, label="Apriori original")
    ax.bar([i + LARGEUR_BARRE / 2 for i in x], [r["tc"] for r in lignes], LARGEUR_BARRE, color=VERT, label="CFP-growth")
    ax.set_xticks(list(x)); ax.set_xticklabels(["%g %%" % (100 * r["ls"]) for r in lignes])
    ax.set_xlabel("seuil unique (= LS)"); ax.set_ylabel("temps (s)"); ax.legend(frameon=False)
    ax.set_title("1. Temps selon le seuil (même seuil pour les deux)", loc="left", fontsize=10)
    rapport.sauver_figure(fig, "1_temps_selon_seuil.png")

    # Graphe 2 : candidats par niveau au seuil LS_FIXE.
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
    """PARTIE 2 : temps selon la taille de la base, avec LS fixe et le même seuil pour les deux."""
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
    rapport.ecrire("\n=== PARTIE 5 : Apriori (minsup = LS = %g %%) contre CFP-growth (LS, beta automatique) ==="
                   % (100 * LS_FIXE))
    ref = ligne_pour(lignes, LS_FIXE)
    c, tc = chrono(lambda: cfp_growth(tickets, LS_FIXE))
    fa, fc = set(ref["a"]["frequents"]), set(c["frequents"])
    rapport.ecrire("Apriori : %d itemsets, %.2f s" % (len(fa), ref["ta"]))
    rapport.ecrire("CFP-growth (beta auto = %s) : %d itemsets, %.2f s" % (etiquette_beta(["%.3f" % b for b in c["betas"]]), len(fc), tc))
    rapport.ecrire("en commun : %d | seulement Apriori : %d | seulement CFP-growth : %d"
                   % (len(fa & fc), len(fa - fc), len(fc - fa)))
    rapport.ecrire("itemsets avec un produit rare (support < %g %%) : Apriori %d | CFP-growth %d"
                   % (100 * RARE, avec_rare(fa, rares), avec_rare(fc, rares)))

    # Graphe 8 : itemsets en commun et différents, par taille.
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
    ax.set_title("8. Itemsets : Apriori (1 seuil) contre CFP-growth (MIS automatique)", loc="left", fontsize=9)
    rapport.sauver_figure(fig, "8_itemsets_communs_et_differents.png")


def verifier_et_mesurer(tickets, rapport, lignes):
    """PARTIE 6 : vérifie que les deux trouvent les mêmes itemsets et compare temps et mémoire."""
    ref = ligne_pour(lignes, LS_FIXE)
    a, c = ref["a"]["frequents"], ref["c"]["frequents"]

    # Graphe 9 : vérification des itemsets par niveau.
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

    # Graphe 10 : comparaison globale (temps et mémoire).
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


# 3. PARTIE 3 : LS varie, Apriori avec minsup = LS contre CFP-Growth avec MIS automatique.
def tester_valeurs_ls(tickets, rares, rapport, lignes):
    rapport.ecrire("\n=== PARTIE 3 : LS varie (beta automatique) ===")
    rapport.ecrire("%-7s | %-28s | %-24s | %s" % ("LS", "Apriori (s | motifs | cand.)", "CFP-growth (s | motifs)",
                                                  "motifs avec produit rare (Apriori | CFP)"))
    E = []
    for ls in LS_LIST:
        ligne = ligne_pour(lignes, ls)
        a, ta = ligne["a"], ligne["ta"]
        c, tc = chrono(lambda: cfp_growth(tickets, ls))
        cand = sum(len(v) for v in a["candidats"].values())
        e = dict(ls=ls, ta=ta, tc=tc, ma=len(a["frequents"]), mc=len(c["frequents"]), cand=cand,
                 ra=avec_rare(a["frequents"], rares), rc=avec_rare(c["frequents"], rares))
        E.append(e)
        rapport.ecrire("%-7s | %6.2f s %7d %9d | %6.2f s %7d       | %d | %d"
                       % ("%g %%" % (100 * ls), ta, e["ma"], cand, tc, e["mc"], e["ra"], e["rc"]))

    xs = [100 * e["ls"] for e in E]
    nom_cfp = "CFP-growth (β automatique)"

    # Graphe 4 : temps selon LS.
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND); style(ax)
    ax.plot(xs, [e["ta"] for e in E], "o-", color=BLEU, label="Apriori (minsup = LS)")
    ax.plot(xs, [e["tc"] for e in E], "s-", color=VERT, label=nom_cfp)
    ax.set_xlabel("LS (%)"); ax.set_ylabel("temps (s)"); ax.invert_xaxis(); ax.legend(frameon=False)
    ax.set_title("4. Temps selon LS", loc="left", fontsize=10)
    rapport.sauver_figure(fig, "4_temps_selon_LS.png")

    # Graphe 5 : motifs et candidats selon LS.
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


# 4. PARTIE 4 : beta calculés par la formule pour chaque LS (plus aucun beta choisi à la main).
def tester_betas_auto(tickets, rares, rapport, lignes):
    rapport.ecrire("\n=== PARTIE 4 : beta calculés automatiquement selon LS ===")
    rapport.ecrire("%-7s | %-20s | %-9s | %-9s | %-9s | %s" % ("LS", "beta f/m/F", "CFP (s)", "itemsets", "Apriori", "motifs MIN-fréquents"))
    E = []
    for ls in LS_LIST:
        c, tc = chrono(lambda: cfp_growth(tickets, ls))
        e = dict(ls=ls, betas=c["betas"], tc=tc, mc=len(c["frequents"]), ma=len(ligne_pour(lignes, ls)["a"]["frequents"]),
                 enum=sum(c["enumeres_par_niveau"].values()), occ=c["occurrences"], noeuds=c["noeuds_arbre"])
        E.append(e)
        rapport.ecrire("%-7s | %-20s | %6.2f s | %-9d | %-9d | %d"
                       % ("%g %%" % (100 * ls), etiquette_beta(["%.3f" % b for b in e["betas"]]), tc, e["mc"], e["ma"], e["enum"]))

    xs = [100 * e["ls"] for e in E]

    # Graphe 6 : beta calculés et itemsets retournés.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=FOND); style(ax1); style(ax2)
    for i, (nom, couleur) in enumerate((("β faible", GRIS), ("β moyen", ORANGE), ("β fort", VERT))):
        ax1.plot(xs, [e["betas"][i] for e in E], "o-", color=couleur, label=nom)
    ax1.plot(xs, [e["ls"] for e in E], "k--", lw=1, label="LS (plancher de β)")
    ax1.set_xlabel("LS (%)"); ax1.set_ylabel("β calculé"); ax1.invert_xaxis(); ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("(a) β calculés par la formule", loc="left", fontsize=10)
    ax2.plot(xs, [e["ma"] for e in E], "o-", color=BLEU, label="Apriori (minsup = LS)")
    ax2.plot(xs, [e["mc"] for e in E], "s--", color=VERT, label="CFP-growth (β automatique)")
    ax2.set_xlabel("LS (%)"); ax2.set_ylabel("itemsets fréquents"); ax2.invert_xaxis(); ax2.legend(frameon=False, fontsize=8)
    ax2.set_title("(b) Itemsets retournés", loc="left", fontsize=10)
    fig.suptitle("6. Bêta automatiques et itemsets selon LS", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    rapport.sauver_figure(fig, "6_beta_automatique.png")

    # Graphe 7 : compression du MIS-tree selon LS.
    rapport.ecrire("\nCompression du MIS-tree selon LS :")
    for e in E:
        rapport.ecrire("   LS %g %% : %d produits dans les tickets -> %d nœuds (÷ %.1f)"
                       % (100 * e["ls"], e["occ"], e["noeuds"], e["occ"] / max(e["noeuds"], 1)))
    x = list(range(len(E)))
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=FOND); style(ax)
    b1 = ax.bar([i - LARGEUR_BARRE / 2 for i in x], [e["occ"] for e in E], LARGEUR_BARRE, color=GRIS, label="Produits dans les tickets")
    b2 = ax.bar([i + LARGEUR_BARRE / 2 for i in x], [e["noeuds"] for e in E], LARGEUR_BARRE, color=VERT, label="Nœuds du MIS-tree")
    etiquettes(ax, b1); etiquettes(ax, b2)
    haut = max(max(e["occ"] for e in E), 1)
    for i, e in enumerate(E):
        ax.text(i, haut * 1.08, "÷ %.1f" % (e["occ"] / max(e["noeuds"], 1)), ha="center", color=VERT, fontweight="bold")
    ax.set_ylim(0, haut * 1.2)
    ax.set_xticks(x); ax.set_xticklabels(["LS = %g %%" % (100 * e["ls"]) for e in E], fontsize=8)
    ax.set_ylabel("Nombre"); ax.legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.1))
    ax.set_title("7. Le MIS-tree compresse la base\n(produits à stocker dans les tickets, après filtrage par MIN, contre nœuds de l'arbre)",
                 loc="left", fontsize=10)
    rapport.sauver_figure(fig, "7_compression_arbre.png")


# 4b. PARTIE 6 : sensibilité au niveau du quantile utilisé pour LS = max(1/racine(N), Q(S)/N).
def tester_niveaux_ls(tickets, rapport):
    N = len(tickets)
    S = list(compter_supports(tickets).values())
    plancher = 1 / math.sqrt(N)
    rapport.ecrire("\n=== PARTIE 6 : LS selon le quantile choisi (plancher 1/racine(N) = %.2f %%) ===" % (100 * plancher))
    rapport.ecrire("%-12s | %-8s | %-9s | %-9s | %-10s | %s" % ("quantile", "LS", "itemsets", "Apriori (s)", "CFP (s)", "candidats Apriori"))
    E = []
    for p in NIVEAUX_LS:
        ls = max(plancher, quantile_interpolation(S, p) / N)
        a, ta = chrono(lambda: apriori_original(tickets, ls))
        c, tc = chrono(lambda: cfp_growth(tickets, ls))
        nom = "P%d%s" % (round(100 * p), " (Q3)" if p == 0.75 else (" (méd.)" if p == 0.5 else ""))
        e = dict(nom=nom, ls=ls, n=len(c["frequents"]), ta=ta, tc=tc)
        E.append(e)
        rapport.ecrire("%-12s | %5.2f %%  | %-9d | %9.2f   | %8.2f   | %d"
                       % (nom, 100 * ls, e["n"], ta, tc, sum(len(v) for v in a["candidats"].values())))

    x = list(range(len(E)))
    etiq = ["%s\nLS = %.1f %%" % (e["nom"], 100 * e["ls"]) for e in E]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=FOND); style(ax1); style(ax2)
    b = ax1.bar(x, [e["n"] for e in E], 0.5, color=VERT)
    etiquettes(ax1, b)
    ax1.set_xticks(x); ax1.set_xticklabels(etiq, fontsize=8); ax1.set_ylabel("itemsets fréquents")
    ax1.set_title("(a) Itemsets selon le quantile de LS", loc="left", fontsize=10)
    b1 = ax2.bar([i - LARGEUR_BARRE / 2 for i in x], [e["ta"] for e in E], LARGEUR_BARRE, color=BLEU, label="Apriori")
    b2 = ax2.bar([i + LARGEUR_BARRE / 2 for i in x], [e["tc"] for e in E], LARGEUR_BARRE, color=VERT, label="CFP-growth")
    etiquettes(ax2, b1, "%.2f"); etiquettes(ax2, b2, "%.2f")
    ax2.set_xticks(x); ax2.set_xticklabels(etiq, fontsize=8); ax2.set_ylabel("temps (s)"); ax2.legend(frameon=False)
    ax2.set_title("(b) Temps d'exécution", loc="left", fontsize=10)
    fig.suptitle("11. Sensibilité au quantile de LS (Q3 = P75 choisi)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    rapport.sauver_figure(fig, "11_sensibilite_niveau_ls.png")


# 5. Mesure du temps (médiane de plusieurs exécutions) et de la mémoire (pic).
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


# 6. Rapport : garde le texte et les graphes puis les enregistre en PNG, PDF et fichier texte.
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


# 7. Programme principal : l'ordre des étapes.
def main():
    rapport = Rapport(DOSSIER_SORTIE)

    tickets, rares = charger_donnees(rapport)
    lignes = comparer_meme_seuil(tickets, rapport)
    tester_taille_base(tickets, rapport)
    tester_valeurs_ls(tickets, rares, rapport, lignes)
    tester_betas_auto(tickets, rares, rapport, lignes)
    comparer_apriori_quartiles(tickets, rares, rapport, lignes)
    tester_niveaux_ls(tickets, rapport)
    verifier_et_mesurer(tickets, rapport, lignes)
    rapport.enregistrer()


if __name__ == "__main__":
    main()