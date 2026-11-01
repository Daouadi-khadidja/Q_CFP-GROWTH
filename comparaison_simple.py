
import argparse
import math
import os
import statistics
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from apriori_original import apriori_original
from cfp_growth import cfp_growth, cfp_growth_seuils
from beta_auto import choisir_beta

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
#  Réglages : valeurs par défaut = protocole du rapport
#    LS : de 5 % à 1,5 %      beta = 0,5 quand il ne varie pas
#    (les valeurs intermédiaires de LS et la liste des beta sont modifiables)
# ---------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--fichier", default=os.path.join(DOSSIER, "Data", "groceries2.csv"))
ap.add_argument("--ls-list", default="0.05,0.04,0.03,0.025,0.02,0.015", help="valeurs de LS (fractions)")
ap.add_argument("--betas", default="0.1,0.2,0.3,0.5,0.7,0.9", help="valeurs de beta testées")
ap.add_argument("--beta", type=float, default=0.5, help="beta utilisé quand il ne varie pas")
ap.add_argument("--ls-fixe", type=float, default=0.015, help="LS utilisé quand il ne varie pas")
ap.add_argument("--tailles", default="1000,2500,5000,7500", help="tailles de base testées (la taille complète s'ajoute)")
ap.add_argument("--budget", type=int, default=2000, help="nombre maximum d'itemsets accepté (beta automatique)")
ap.add_argument("--cible", type=float, default=0.95, help="proportion minimale de paires à lift > 1 (beta automatique)")
ap.add_argument("--rare", type=float, default=0.05, help="un produit est rare si son support est < cette fraction")
ap.add_argument("--repetitions", type=int, default=3, help="exécutions par mesure (on garde la médiane)")
args = ap.parse_args()

LS_LIST = sorted((float(x) for x in args.ls_list.split(",")), reverse=True)
BETAS = sorted(float(x) for x in args.betas.split(","))
TAILLES = [int(x) for x in args.tailles.split(",")]
LS_FIXE, BETA, BUDGET, CIBLE = args.ls_fixe, args.beta, args.budget, args.cible

SORTIE = os.path.join(DOSSIER, "resultats_simple")
os.makedirs(SORTIE, exist_ok=True)
lignes, figures = [], []

FOND = "#fdfcfa"
BLEU, VERT, GRIS, ORANGE = "#2878d6", "#2e9f6e", "#9b9a94", "#e8742f"


def ecrire(txt=""):
    print(txt)
    lignes.append(txt)


def chrono(f):
    temps = []
    for _ in range(max(1, args.repetitions)):
        d = time.perf_counter()
        r = f()
        temps.append(time.perf_counter() - d)
    return r, statistics.median(temps)


def pic_memoire(f):
    tracemalloc.start()
    f()
    p = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return p


def sauver(fig, nom):
    fig.tight_layout()
    fig.savefig(os.path.join(SORTIE, nom), dpi=130)
    figures.append(fig)


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


def seuil_unique(T, minsup):
    s = math.ceil(round(minsup * len(T), 9))
    return {p: s for t in T for p in t}


def avec_rare(frequents, rares):
    return sum(1 for g in frequents if any(p in rares for p in g))


tickets = []
for l in open(args.fichier, encoding="utf-8"):
    p = sorted({x.strip() for x in l.split(",") if x.strip()})
    if p:
        tickets.append(p)
N = len(tickets)
nb_prod = {}
for t in tickets:
    for p in t:
        nb_prod[p] = nb_prod.get(p, 0) + 1
rares = {p for p in nb_prod if nb_prod[p] < args.rare * N}
ecrire("DONNÉES : %d tickets, %d produits (rares, support < %g %% : %d)" % (N, len(nb_prod), 100 * args.rare, len(rares)))
ecrire("RÉGLAGES : LS = %s | beta = %s | beta fixe = %g | LS fixe = %g %% | médiane de %d exécutions"
       % ([round(100 * x, 2) for x in LS_LIST], BETAS, BETA, 100 * LS_FIXE, args.repetitions))

# =================================================================
#  PARTIE 1 : même seuil unique pour les deux (LS varie)
# =================================================================
ecrire("\n=== PARTIE 1 : même seuil unique pour les deux (comparaison équitable) ===")
ecrire("%-8s | %-24s | %-26s | mêmes itemsets" % ("seuil", "Apriori (s | candidats)", "CFP-growth (s | motifs)"))
p1 = []
for ls in LS_LIST:
    a, ta = chrono(lambda: apriori_original(tickets, ls))
    c, tc = chrono(lambda: cfp_growth_seuils(tickets, seuil_unique(tickets, ls)))
    nca = sum(len(v) for v in a["candidats"].values())
    nce = sum(c["enumeres_par_niveau"].values())
    meme = set(a["frequents"]) == set(c["frequents"])
    p1.append(dict(ls=ls, ta=ta, tc=tc, nca=nca, nce=nce, meme=meme, a=a, c=c))
    ecrire("%-8s | %6.2f s | %6d cand. | %6.2f s | %6d motifs | %s"
           % ("%g %%" % (100 * ls), ta, nca, tc, nce, meme))

w = 0.38
fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND); style(ax)
x = range(len(p1))
ax.bar([i - w / 2 for i in x], [r["ta"] for r in p1], w, color=BLEU, label="Apriori original")
ax.bar([i + w / 2 for i in x], [r["tc"] for r in p1], w, color=VERT, label="CFP-growth")
ax.set_xticks(list(x)); ax.set_xticklabels(["%g %%" % (100 * r["ls"]) for r in p1])
ax.set_xlabel("seuil unique (= LS)"); ax.set_ylabel("temps (s)"); ax.legend(frameon=False)
ax.set_title("1. Temps selon le seuil (même seuil pour les deux)", loc="left", fontsize=10)
sauver(fig, "1_temps_selon_seuil.png")

ref = min(p1, key=lambda r: abs(r["ls"] - LS_FIXE))
a_ref, c_ref = ref["a"], ref["c"]
niv = sorted(set(a_ref["candidats"]) | set(c_ref["enumeres_par_niveau"]))
fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND); style(ax)
ax.bar([i - w / 2 for i in range(len(niv))], [len(a_ref["candidats"].get(k, [])) for k in niv], w,
       color=BLEU, label="Apriori : candidats testés")
ax.bar([i + w / 2 for i in range(len(niv))], [c_ref["enumeres_par_niveau"].get(k, 0) for k in niv], w,
       color=VERT, label="CFP-growth : motifs trouvés (0 candidat généré)")
ax.set_xticks(range(len(niv))); ax.set_xticklabels(["taille %d" % k for k in niv])
ax.set_ylabel("nombre"); ax.legend(frameon=False, fontsize=8)
ax.set_title("2. Candidats par niveau (seuil %g %%)" % (100 * ref["ls"]), loc="left", fontsize=10)
sauver(fig, "2_candidats_par_niveau.png")

# =================================================================
#  PARTIE 2 : taille de la base (LS fixe)
# =================================================================
ecrire("\n=== PARTIE 2 : temps selon la taille de la base (seuil %g %%) ===" % (100 * LS_FIXE))
ns, ta_l, tc_l = [], [], []
for n in [t for t in TAILLES if t < N] + [N]:
    T = tickets[:n]
    _, ta = chrono(lambda: apriori_original(T, LS_FIXE))
    _, tc = chrono(lambda: cfp_growth_seuils(T, seuil_unique(T, LS_FIXE)))
    ns.append(n); ta_l.append(ta); tc_l.append(tc)
    ecrire("N = %5d : Apriori %.2f s | CFP-growth %.2f s | rapport x%.1f" % (n, ta, tc, ta / tc))
fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND); style(ax)
ax.plot(ns, ta_l, "o-", color=BLEU, label="Apriori original")
ax.plot(ns, tc_l, "s-", color=VERT, label="CFP-growth")
ax.set_xlabel("nombre de tickets"); ax.set_ylabel("temps (s)"); ax.legend(frameon=False)
ax.set_title("3. Temps selon la taille de la base (seuil %g %%)" % (100 * LS_FIXE), loc="left", fontsize=10)
sauver(fig, "3_temps_selon_taille.png")

# =================================================================
#  PARTIE 3 : LS varie  (Apriori : minsup = LS | CFP-growth : MIS, beta fixe)
# =================================================================
ecrire("\n=== PARTIE 3 : LS varie (beta = %g) ===" % BETA)
ecrire("%-7s | %-28s | %-24s | %s" % ("LS", "Apriori (s | motifs | cand.)", "CFP-growth (s | motifs)", "motifs avec produit rare (Apriori | CFP)"))
E1 = []
for ls in LS_LIST:
    a, ta = chrono(lambda: apriori_original(tickets, ls))
    c, tc = chrono(lambda: cfp_growth(tickets, ls, BETA))
    cand = sum(len(v) for v in a["candidats"].values())
    E1.append(dict(ls=ls, ta=ta, tc=tc, ma=len(a["frequents"]), mc=len(c["frequents"]), cand=cand,
                   ra=avec_rare(a["frequents"], rares), rc=avec_rare(c["frequents"], rares)))
    e = E1[-1]
    ecrire("%-7s | %6.2f s %7d %9d | %6.2f s %7d       | %d | %d"
           % ("%g %%" % (100 * ls), ta, e["ma"], cand, tc, e["mc"], e["ra"], e["rc"]))

xs = [100 * e["ls"] for e in E1]
fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND); style(ax)
ax.plot(xs, [e["ta"] for e in E1], "o-", color=BLEU, label="Apriori (minsup = LS)")
ax.plot(xs, [e["tc"] for e in E1], "s-", color=VERT, label="CFP-growth (β = %g)" % BETA)
ax.set_xlabel("LS (%)"); ax.set_ylabel("temps (s)"); ax.invert_xaxis(); ax.legend(frameon=False)
ax.set_title("9. Temps selon LS", loc="left", fontsize=10)
sauver(fig, "9_temps_selon_LS.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=FOND); style(ax1); style(ax2)
ax1.plot(xs, [e["ma"] for e in E1], "o-", color=BLEU, label="Apriori")
ax1.plot(xs, [e["mc"] for e in E1], "s-", color=VERT, label="CFP-growth (β = %g)" % BETA)
ax1.set_xlabel("LS (%)"); ax1.set_ylabel("motifs fréquents"); ax1.invert_xaxis(); ax1.legend(frameon=False)
ax1.set_title("(a) Motifs fréquents", loc="left", fontsize=10)
ax2.plot(xs, [e["cand"] for e in E1], "o-", color=BLEU, label="Apriori")
ax2.plot(xs, [0] * len(E1), "s-", color=VERT, label="CFP-growth (aucun candidat)")
ax2.set_xlabel("LS (%)"); ax2.set_ylabel("candidats générés (cumulés)"); ax2.invert_xaxis(); ax2.legend(frameon=False)
ax2.set_title("(b) Candidats générés", loc="left", fontsize=10)
fig.suptitle("10. Motifs et candidats selon LS", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
sauver(fig, "10_motifs_et_candidats_selon_LS.png")

# =================================================================
#  PARTIE 4 : beta varie (LS fixe)
# =================================================================
ecrire("\n=== PARTIE 4 : beta varie (LS = %g %%, Apriori : minsup = LS) ===" % (100 * LS_FIXE))
a_fix, ta_fix = chrono(lambda: apriori_original(tickets, LS_FIXE))
ecrire("Apriori (ne dépend pas de beta) : %.2f s, %d itemsets" % (ta_fix, len(a_fix["frequents"])))
ecrire("%-6s | %-9s | %-9s | %-22s | %s" % ("beta", "CFP (s)", "itemsets", "motifs MIN-fréquents", "avec produit rare"))
E2 = []
for b in BETAS:
    c, tc = chrono(lambda: cfp_growth(tickets, LS_FIXE, b))
    E2.append(dict(beta=b, tc=tc, mc=len(c["frequents"]), enum=sum(c["enumeres_par_niveau"].values()),
                   rc=avec_rare(c["frequents"], rares), occ=c["occurrences"], noeuds=c["noeuds_arbre"]))
    e = E2[-1]
    ecrire("%-6g | %6.2f s | %-9d | %-22d | %d" % (b, tc, e["mc"], e["enum"], e["rc"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=FOND); style(ax1); style(ax2)
ax1.plot(BETAS, [e["tc"] for e in E2], "s-", color=VERT, label="CFP-growth")
ax1.axhline(ta_fix, color=BLEU, ls="--", label="Apriori (indépendant de β)")
ax1.set_xlabel("β"); ax1.set_ylabel("temps (s)"); ax1.legend(frameon=False)
ax1.set_title("(a) Temps selon β", loc="left", fontsize=10)
ax2.plot(BETAS, [e["mc"] for e in E2], "s-", color=VERT, label="CFP-growth : itemsets retournés")
ax2.plot(BETAS, [e["enum"] for e in E2], "d--", color=ORANGE, label="CFP-growth : motifs MIN-fréquents explorés")
ax2.axhline(len(a_fix["frequents"]), color=BLEU, ls="--", label="Apriori (minsup = LS)")
ax2.set_xlabel("β"); ax2.set_ylabel("nombre d'itemsets"); ax2.legend(frameon=False, fontsize=8)
ax2.set_title("(b) Itemsets selon β", loc="left", fontsize=10)
fig.suptitle("11. Influence de β (LS = %g %%)" % (100 * LS_FIXE), x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
sauver(fig, "11_influence_beta.png")

# graphe 4 : compression du MIS-tree selon beta
ecrire("\nCompression du MIS-tree selon beta (LS = %g %%) :" % (100 * LS_FIXE))
for e in E2:
    ecrire("   beta %g : %d produits dans les tickets -> %d nœuds (÷ %.1f)" % (e["beta"], e["occ"], e["noeuds"], e["occ"] / e["noeuds"]))
fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=FOND); style(ax)
b1 = ax.bar([i - w / 2 for i in range(len(BETAS))], [e["occ"] for e in E2], w, color=GRIS, label="Produits dans les tickets")
b2 = ax.bar([i + w / 2 for i in range(len(BETAS))], [e["noeuds"] for e in E2], w, color=VERT, label="Nœuds du MIS-tree")
etiquettes(ax, b1); etiquettes(ax, b2)
top = max(e["occ"] for e in E2)
for i, e in enumerate(E2):
    ax.text(i, top * 1.08, "÷ %.1f" % (e["occ"] / e["noeuds"]), ha="center", color=VERT, fontweight="bold")
ax.set_ylim(0, top * 1.2)
ax.set_xticks(range(len(BETAS))); ax.set_xticklabels(["β = %g" % b for b in BETAS])
ax.set_ylabel("Nombre"); ax.legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.1))
ax.set_title("4. Le MIS-tree compresse la base\n(produits à stocker dans les tickets, après filtrage par MIN, contre nœuds de l'arbre)",
             loc="left", fontsize=10)
sauver(fig, "4_compression_arbre.png")

# =================================================================
#  PARTIE 5 : choix automatique de beta (valeurs données par le client)
# =================================================================
ecrire("\n=== PARTIE 5 : choix automatique de beta (LS=%g %%, budget=%d itemsets, cible=%d %% de lift>1) ==="
       % (100 * LS_FIXE, BUDGET, round(100 * CIBLE)))
ecrire("valeurs de beta données par le client : %s" % BETAS)
beta_choisi, tab = choisir_beta(tickets, LS_FIXE, BETAS, BUDGET, CIBLE)
ecrire("%-6s | %-9s | %-7s | %s" % ("beta", "itemsets", "paires", "paires lift>1"))
for l in tab:
    q = "-" if l["lift_sup_1"] is None else "%.1f %%" % (100 * l["lift_sup_1"])
    ecrire("%-6g | %-9d | %-7d | %s %s" % (l["beta"], l["itemsets"], l["paires"], q, "<-- choisi" if l["beta"] == beta_choisi else ""))

fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=FOND); style(ax)
ax.plot([l["beta"] for l in tab], [l["itemsets"] for l in tab], "o-", color=BLEU)
ax.axhline(BUDGET, color=BLEU, ls="--", lw=1)
ax.set_xlabel("beta"); ax.set_ylabel("itemsets (bleu), budget en pointillés", color=BLEU)
ax2 = ax.twinx()
ok = [l for l in tab if l["lift_sup_1"] is not None]
ax2.plot([l["beta"] for l in ok], [100 * l["lift_sup_1"] for l in ok], "s-", color="tab:red")
ax2.axhline(100 * CIBLE, color="tab:red", ls="--", lw=1)
ax2.set_ylabel("% de paires à lift > 1 (rouge), cible en pointillés", color="tab:red")
ax.axvline(beta_choisi, color="gray", ls=":")
ax.set_title("5. Choix automatique de beta (choisi : %g)" % beta_choisi, loc="left", fontsize=10)
sauver(fig, "5_choix_beta.png")

# =================================================================
#  PARTIE 6 : Apriori (seuil unique) contre CFP-growth (seuil par produit)
# =================================================================
ecrire("\n=== PARTIE 6 : Apriori (minsup = LS = %g %%) contre CFP-growth (LS, beta = %g) ===" % (100 * LS_FIXE, beta_choisi))
c6, tc6 = chrono(lambda: cfp_growth(tickets, LS_FIXE, beta_choisi))
fa, fc = set(a_fix["frequents"]), set(c6["frequents"])
ecrire("Apriori : %d itemsets, %.2f s" % (len(fa), ta_fix))
ecrire("CFP-growth (beta = %g) : %d itemsets, %.2f s" % (beta_choisi, len(fc), tc6))
ecrire("en commun : %d | seulement Apriori : %d | seulement CFP-growth : %d" % (len(fa & fc), len(fa - fc), len(fc - fa)))
ecrire("itemsets avec un produit rare (support < %g %%) : Apriori %d | CFP-growth %d"
       % (100 * args.rare, avec_rare(fa, rares), avec_rare(fc, rares)))

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
ax.set_title("6. Itemsets : Apriori (1 seuil) contre CFP-growth (seuil par produit, β = %g)" % beta_choisi, loc="left", fontsize=9)
sauver(fig, "6_itemsets_communs_et_differents.png")

# graphe 7 : vérification (même seuil unique)
niv = sorted({len(g) for g in a_ref["frequents"]} | {len(g) for g in c_ref["frequents"]})
na = [sum(1 for g in a_ref["frequents"] if len(g) == k) for k in niv]
nc = [sum(1 for g in c_ref["frequents"] if len(g) == k) for k in niv]
identiques = set(a_ref["frequents"]) == set(c_ref["frequents"])
fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=FOND); style(ax)
b1 = ax.bar([i - w / 2 for i in range(len(niv))], na, w, color=BLEU, label="Apriori (seuil %g %%)" % (100 * ref["ls"]))
b2 = ax.bar([i + w / 2 for i in range(len(niv))], nc, w, color=VERT, label="CFP-growth (même seuil)")
etiquettes(ax, b1); etiquettes(ax, b2)
ax.set_xticks(range(len(niv))); ax.set_xticklabels(["Niveau %d" % k for k in niv])
ax.set_ylabel("Itemsets fréquents"); ax.legend(frameon=False)
ax.set_title("7. Vérification : CFP-growth trouve les mêmes itemsets qu'Apriori\n%d tickets. Itemsets identiques : %s (%d itemsets au total)"
             % (N, "OUI" if identiques else "NON", len(a_ref["frequents"])), loc="left", fontsize=10)
sauver(fig, "7_verification_cfp.png")

# graphe 8 : comparaison des deux algorithmes (même seuil unique)
mem_a = pic_memoire(lambda: apriori_original(tickets, ref["ls"]))
mem_c = pic_memoire(lambda: cfp_growth_seuils(tickets, seuil_unique(tickets, ref["ls"])))
valeurs = {
    "Itemsets fréquents trouvés": (len(a_ref["frequents"]), len(c_ref["frequents"]), "%d"),
    "Candidats testés (Apriori)\nou motifs fréquents (CFP)": (ref["nca"], ref["nce"], "%d"),
    "Passages sur la base\n(un par niveau, ou 2 pour CFP)": (len(a_ref["candidats"]), c_ref["passages"], "%d"),
    "Temps d'exécution (s)": (ref["ta"], ref["tc"], "%.2f"),
    "Mémoire utilisée, pic (Mo)": (mem_a, mem_c, "%.1f"),
}
ecrire("\n=== COMPARAISON DES DEUX ALGORITHMES (seuil unique %g %% pour les deux) ===" % (100 * ref["ls"]))
fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), facecolor=FOND)
for ax, (titre, (va, vc, fmt)) in zip(axes.flat, valeurs.items()):
    style(ax)
    barres = ax.bar(["Apriori", "CFP-\nGrowth"], [va, vc], color=[BLEU, VERT], width=0.55)
    etiquettes(ax, barres, fmt)
    ax.set_ylim(0, max(va, vc) * 1.18)
    ax.set_title(titre, loc="left", fontsize=9, fontweight="bold")
    ecrire("%-52s Apriori %s | CFP-growth %s" % (titre.replace("\n", " "), fmt % va, fmt % vc))
ax = axes.flat[5]; ax.axis("off")
ax.text(0, 0.95, "Comment lire ces graphes", fontweight="bold", va="top", fontsize=10)
ax.text(0, 0.82, "Apriori et CFP-Growth utilisent le même\nseuil unique : mêmes itemsets.\n\n"
        "CFP-Growth ne génère pas de candidats :\nil énumère les motifs fréquents dans l'arbre.\n\n"
        "Apriori relit la base à chaque niveau.\nCFP-Growth la lit 2 fois.",
        va="top", fontsize=9, color="#444")
fig.suptitle("8. Comparaison des deux algorithmes", x=0.01, ha="left", fontweight="bold", fontsize=13)
fig.text(0.01, 0.925, "%d tickets. Apriori et CFP-Growth : seuil unique = %g %%" % (N, 100 * ref["ls"]), fontsize=9, color="#555")
fig.tight_layout(rect=(0, 0, 1, 0.9))
sauver(fig, "8_comparaison_deux_algorithmes.png")

# =================================================================
#  Sorties
# =================================================================
figures.sort(key=lambda f: int(f.axes[0].get_title(loc="left").split(".")[0]) if f.axes[0].get_title(loc="left")[:1].isdigit() else 99)
with PdfPages(os.path.join(SORTIE, "graphes_simple.pdf")) as pdf:
    for f in figures:
        pdf.savefig(f)
open(os.path.join(SORTIE, "resultats_simple.txt"), "w", encoding="utf-8").write("\n".join(lignes))
print("\nÉcrit dans le dossier resultats_simple : 11 graphes PNG, graphes_simple.pdf, resultats_simple.txt")
