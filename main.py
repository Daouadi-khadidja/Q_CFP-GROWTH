# Lance CFP-Growth sur le dataset diabète puis génère les règles SI ... ALORS ... (python main.py).
import csv
import os
import time
from itertools import combinations

from cfp_growth import cfp_growth

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# RÉGLAGES : les seules valeurs à modifier avant de relancer.
FICHIER = os.path.join(DOSSIER, "Data", "diabetic_data_apriori.csv")   # dataset
DOSSIER_SORTIE = os.path.join(DOSSIER, "resultats_diabete")            # dossier des résultats

LS = 0.10                    # support minimum le plus bas (10 % des patients)
BETAS = (0.2, 0.5, 0.8)      # beta croissants : produits faibles, moyens, forts

CONFIANCE_MIN = 0.6          # confiance minimale des règles
LIFT_MIN = 1.0               # lift minimal des règles
CONCLUSION_MAX = 1           # nombre max d'items dans la conclusion (0 = sans limite)
CONCLUSION_CONTIENT = ""     # filtre : la conclusion doit contenir ce mot

SEPARATEUR = "  ET  "        # séparateur entre items
MAX_LIGNES_ECRAN = 15        # lignes affichées à l'écran


# 1. Lit le tableau 0/1 : un patient = un ticket = la liste des colonnes à 1.
def lire_tickets(fichier):
    tickets = []
    with open(fichier, encoding="utf-8") as f:
        lecteur = csv.reader(f)
        noms = next(lecteur)
        for valeurs in lecteur:
            ticket = sorted(noms[i] for i, v in enumerate(valeurs) if v in ("1", "True"))
            if ticket:
                tickets.append(ticket)
    return tickets


# 2. Lance CFP-Growth (MIS par quartiles) et affiche un résumé.
def lancer_cfp_growth(tickets):
    debut = time.perf_counter()
    res = cfp_growth(tickets, LS, BETAS)
    duree = time.perf_counter() - debut

    N = len(tickets)
    q1, q2 = res["coupures"]
    noms = ("faible", "moyenne", "forte")
    print("Quartiles : Q1 = %.0f patients (%.1f %%), Q2 = %.0f patients (%.1f %%)"
          % (q1, 100 * q1 / N, q2, 100 * q2 / N))
    for categorie in range(3):
        n = sum(1 for c in res["categories"].values() if c == categorie)
        print("   catégorie %-8s : %3d items, beta = %g" % (noms[categorie], n, BETAS[categorie]))
    print("Itemsets fréquents : %d (%.1f s)" % (len(res["frequents"]), duree))
    return res["frequents"]


# 3. Règles d'association : support = Sup(G+C)/N, confiance = Sup(G+C)/Sup(G), lift = confiance / (Sup(C)/N).
def preparer_bitsets(tickets):
    """Chaque item devient un grand entier binaire (1 bit par ticket) pour compter vite."""
    bits = {}
    for i, ticket in enumerate(tickets):
        for item in set(ticket):
            bits[item] = bits.get(item, 0) | (1 << i)
    return bits


def calculer_support(items, bitsets, deja_calcule):
    """Nombre de tickets contenant tous les items (résultats mis en cache)."""
    cle = tuple(sorted(items))
    if cle not in deja_calcule:
        nombre = bitsets[cle[0]]
        for item in cle[1:]:
            nombre &= bitsets[item]
        deja_calcule[cle] = nombre.bit_count()
    return deja_calcule[cle]


def generer_regles(frequents, N, bitsets):
    """Fabrique les règles SI ... ALORS ... qui respectent confiance, lift et conclusion, triées par lift."""
    deja_calcule = dict(frequents)
    mot = CONCLUSION_CONTIENT.lower()
    regles = []
    for groupe, sup in frequents.items():
        taille = len(groupe)
        if taille < 2:
            continue
        limite = taille - 1 if CONCLUSION_MAX == 0 else min(CONCLUSION_MAX, taille - 1)
        for k in range(1, limite + 1):
            for conclusion in combinations(groupe, k):
                if mot and not all(mot in c.lower() for c in conclusion):
                    continue
                gauche = [i for i in groupe if i not in conclusion]
                confiance = sup / calculer_support(gauche, bitsets, deja_calcule)
                if confiance < CONFIANCE_MIN:
                    continue
                lift = confiance / (calculer_support(conclusion, bitsets, deja_calcule) / N)
                if lift < LIFT_MIN:
                    continue
                regles.append({"si": SEPARATEUR.join(gauche), "alors": SEPARATEUR.join(conclusion),
                               "nb_patients": sup, "support_%": round(100 * sup / N, 2),
                               "confiance_%": round(100 * confiance, 1), "lift": round(lift, 2)})
    regles.sort(key=lambda r: (-r["lift"], -r["confiance_%"]))
    return regles


# 4. Sauvegarde en tableaux : colonne = (clé du dictionnaire, titre affiché, est-ce un nombre ?).
COLONNES_ITEMSETS = [("itemset", "Itemset", False), ("taille", "Taille", True),
                     ("nb_patients", "Nb patients", True), ("support_%", "Support (%)", True)]

COLONNES_REGLES = [("si", "SI (conditions)", False), ("alors", "ALORS (conclusion)", False),
                   ("nb_patients", "Nb patients", True), ("support_%", "Support (%)", True),
                   ("confiance_%", "Confiance (%)", True), ("lift", "Lift", True)]


def construire_itemsets(frequents, N):
    """Lignes du tableau des itemsets, triées par taille puis par fréquence décroissante."""
    lignes = [{"itemset": SEPARATEUR.join(g), "taille": len(g), "nb_patients": s, "support_%": round(100 * s / N, 2)}
              for g, s in frequents.items()]
    return sorted(lignes, key=lambda l: (l["taille"], -l["nb_patients"]))


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


# Programme principal : lire les données, CFP-Growth, règles, sauvegarde.
def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    tickets = lire_tickets(FICHIER)
    N = len(tickets)
    nb_items = len({p for t in tickets for p in t})
    print("Données : %d patients, %d items, %.1f items par patient en moyenne"
          % (N, nb_items, sum(len(t) for t in tickets) / N))
    print("LS = %g, beta = %s\n" % (LS, BETAS))

    frequents = lancer_cfp_growth(tickets)

    regles = generer_regles(frequents, N, preparer_bitsets(tickets))

    sauver("regles", "RÈGLES D'ASSOCIATION  (confiance >= %g%%, triées par lift)  -  %d règles"
           % (100 * CONFIANCE_MIN, len(regles)), COLONNES_REGLES, regles)
    sauver("itemsets", "ITEMSETS FRÉQUENTS  -  %d itemsets" % len(frequents),
           COLONNES_ITEMSETS, construire_itemsets(frequents, N))

    print("\nFichiers enregistrés dans :", DOSSIER_SORTIE, "(regles.txt, itemsets.txt, + .csv)")


if __name__ == "__main__":
    main()