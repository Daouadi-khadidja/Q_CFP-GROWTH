# Chef d'orchestre (python main.py) :
#   1. lit le dataset solaire             2. mis.py calcule les MIS et écrit mis.txt
#   3. cfp_growth.py lit mis.txt          4. regles.py génère les règles, puis on sauvegarde tout.
# Avant la première fois : lancer python nettoyer.py (il crée Data/Lim_solar_data_net.csv).
import csv
import os
import time

from cfp_growth import cfp_growth
from mis import NOMS_CATEGORIES, calculer_ls_auto, calculer_mis, compter_supports, ecrire_mis_txt
from regles import generer_regles, preparer_bitsets

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# RÉGLAGES : les seules valeurs à modifier avant de relancer.
FICHIER = os.path.join(DOSSIER, "Data", "Lim_solar_data_net.csv")      # dataset solaire nettoyé
DOSSIER_SORTIE = os.path.join(DOSSIER, "resultats_solaire")            # dossier des résultats
FICHIER_MIS = os.path.join(DOSSIER_SORTIE, "mis.txt")                  # fichier écrit par mis.py, lu par cfp_growth.py

LS = None                    # None = LS automatique : max(1/racine(N), Q3/N) ; sinon une fraction (0.10 = 10 %)

CONFIANCE_MIN = 0.6          # confiance minimale des règles
LIFT_MIN = 1.0               # lift minimal des règles
CONCLUSION_MAX = 0          # nombre max d'items dans la conclusion (0 = sans limite)
CONCLUSION_CONTIENT = ""     # filtre : la conclusion doit contenir ce mot

SEPARATEUR = "  ET  "        # séparateur entre items
MAX_LIGNES_ECRAN = 15        # lignes affichées à l'écran


# 1. Lit le fichier : une ligne = un ticket = la liste de ses items.
#    Colonne 0/1 -> item = nom de la colonne. Colonne avec des tranches -> item = "colonne=tranche".
def nettoyer(valeur):
    """Enlève les guillemets et antislashs laissés par l'export Weka : '\\'(1.2-3.4]\\'' devient (1.2-3.4]."""
    return valeur.replace("\\", "").replace("'", "").strip()


def lire_tickets(fichier):
    tickets = []
    with open(fichier, encoding="utf-8-sig") as f:
        lecteur = csv.reader(f)
        noms = [nettoyer(n) for n in next(lecteur)]
        for valeurs in lecteur:
            ticket = []
            for nom, v in zip(noms, valeurs):
                v = nettoyer(v)
                if v in ("0", "False", ""):
                    continue
                ticket.append(nom if v in ("1", "True") else "%s=%s" % (nom, v))
            if ticket:
                tickets.append(sorted(ticket))
    return tickets


# 2. Lance CFP-Growth (il lit mis.txt) et affiche un résumé.
def lancer_cfp_growth(tickets, m):
    debut = time.perf_counter()
    res = cfp_growth(tickets, FICHIER_MIS)
    duree = time.perf_counter() - debut

    N = len(tickets)
    q1, q3 = m["coupures"]
    print("Quartiles : Q1 = %.0f lignes (%.1f %%), Q3 = %.0f lignes (%.1f %%)"
          % (q1, 100 * q1 / N, q3, 100 * q3 / N))
    for categorie in range(3):
        n = sum(1 for c in m["categories"].values() if c == categorie)
        print("   catégorie %-8s : %3d items, beta = %.3f" % (NOMS_CATEGORIES[categorie], n, m["betas"][categorie]))
    print("Itemsets fréquents : %d (%.1f s)" % (len(res["frequents"]), duree))
    return res["frequents"]


# 3. Sauvegarde en tableaux : colonne = (clé du dictionnaire, titre affiché, est-ce un nombre ?).
COLONNES_ITEMSETS = [("itemset", "Itemset", False), ("taille", "Taille", True),
                     ("nb_lignes", "Nb lignes", True), ("support_%", "Support (%)", True)]

COLONNES_REGLES = [
    ("si", "SI (conditions)", False),
    ("alors", "ALORS (conclusion)", False),
    ("support", "Support relatif", True),
    ("confiance", "Confiance", True),
    ("lift", "Lift", True)
]


def construire_itemsets(frequents, N):
    """Lignes du tableau des itemsets, triées par taille puis par fréquence décroissante."""
    lignes = [{"itemset": SEPARATEUR.join(g), "taille": len(g), "nb_lignes": s, "support_%": round(100 * s / N, 2)}
              for g, s in frequents.items()]
    return sorted(lignes, key=lambda l: (l["taille"], -l["nb_lignes"]))


def tableau_texte(titre, colonnes, lignes, max_lignes=None, largeur_texte=None):
    """Fabrique un tableau texte aligné (textes coupés avec … si largeur_texte est donnée)."""
    def case(valeur, nombre):
        valeur = str(valeur)
        return valeur if nombre or not largeur_texte or len(valeur) <= largeur_texte else valeur[:largeur_texte - 1] + "…"

    cases = [[case(l[cle], nombre) for cle, _, nombre in colonnes] for l in lignes[:max_lignes]]
    largeurs = [max([len(titre_col)] + [len(c[j]) for c in cases]) for j, (_, titre_col, _) in enumerate(colonnes)]

    def ligne(valeurs):
        return "| " + " | ".join(v.rjust(w) if colonnes[j][2] else v.ljust(w)
                                 for j, (v, w) in enumerate(zip(valeurs, largeurs))) + " |"

    trait = "+-" + "-+-".join("-" * w for w in largeurs) + "-+"
    sortie = [titre, trait, ligne([t for _, t, _ in colonnes]), trait] + [ligne(c) for c in cases] + [trait]
    if max_lignes and len(lignes) > max_lignes:
        sortie.append("... %d lignes de plus dans les fichiers .txt / .csv" % (len(lignes) - max_lignes))
    return "\n".join(sortie)


def sauver(nom, titre, colonnes, lignes):
    """Écrit le tableau en .txt et .csv puis l'affiche à l'écran."""
    base = os.path.join(DOSSIER_SORTIE, nom)
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(tableau_texte(titre, colonnes, lignes) + "\n")
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow([t for _, t, _ in colonnes])
        ecrivain.writerows([[l[cle] for cle, _, _ in colonnes] for l in lignes])
    print("\n" + tableau_texte(titre, colonnes, lignes, MAX_LIGNES_ECRAN, largeur_texte=48))


# 4. Programme principal.
def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    if not os.path.exists(FICHIER):
        print("Fichier introuvable :", FICHIER)
        print("Lance d'abord : python nettoyer.py")
        return

    # a) Lire les données.
    tickets = lire_tickets(FICHIER)
    N = len(tickets)
    nb_items = len({p for t in tickets for p in t})
    print("Données : %d lignes, %d items, %.1f items par ligne en moyenne"
          % (N, nb_items, sum(len(t) for t in tickets) / N))
    ls = LS if LS is not None else calculer_ls_auto(compter_supports(tickets), N)
    print("LS = %.2f %% (%s)\n"
          % (100 * ls, "choisi à la main" if LS is not None else "calculé : max(1/racine(N), Q3/N)"))

    # b) mis.py calcule les MIS et les écrit dans mis.txt.
    m = calculer_mis(tickets, ls)
    ecrire_mis_txt(m, FICHIER_MIS)
    print("MIS écrits dans :", FICHIER_MIS, "\n")

    # c) cfp_growth.py lit mis.txt et trouve les itemsets fréquents.
    frequents = lancer_cfp_growth(tickets, m)

    # d) Règles d'association.
    regles = generer_regles(frequents, N, preparer_bitsets(tickets),
                            confiance_min=CONFIANCE_MIN, lift_min=LIFT_MIN,
                            conclusion_max=CONCLUSION_MAX, conclusion_contient=CONCLUSION_CONTIENT,
                            separateur=SEPARATEUR)

    sauver("regles", "RÈGLES D'ASSOCIATION  (confiance >= %g%%, triées par lift)  -  %d règles"
           % (100 * CONFIANCE_MIN, len(regles)), COLONNES_REGLES, regles)
    sauver("itemsets", "ITEMSETS FRÉQUENTS  -  %d itemsets" % len(frequents),
           COLONNES_ITEMSETS, construire_itemsets(frequents, N))

    print("\nFichiers enregistrés dans :", DOSSIER_SORTIE, "(mis.txt, regles.txt, itemsets.txt, + .csv)")


if __name__ == "__main__":
    main()