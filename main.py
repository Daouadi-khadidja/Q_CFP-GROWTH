# =====================================================================
#  MAIN.PY  -  CFP-Growth + règles d'association sur le dataset diabète
#
#  POUR LANCER :   python main.py
#  POUR CHANGER LES RÉGLAGES : modifier les valeurs juste en dessous (RÉGLAGES), puis relancer.
#
#  Ce que fait ce fichier, dans l'ordre :
#     1. Read the diabetes dataset     -> lire_tickets
#     2. Run CFP-Growth                -> lancer_cfp_growth  (MIS par quartiles)
#     3. Generate association rules    -> generer_regles
#     4. Save results as tables        -> fichiers .txt (tableaux lisibles) + .csv
#
#  Une règle s'écrit :  SI (partie gauche)  ALORS (conclusion)
#     support   = Sup(gauche + conclusion) / N
#     confiance = Sup(gauche + conclusion) / Sup(gauche)
#     lift      = confiance / (Sup(conclusion) / N)
#                 lift > 1 : ensemble PLUS que par hasard | 1 : aucun lien | < 1 : plutôt opposés
# =====================================================================
import csv
import os
import time
from itertools import combinations

from cfp_growth import cfp_growth

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
#  RÉGLAGES  (les seules lignes à modifier)
# =====================================================================
FICHIER = os.path.join(DOSSIER, "Data", "diabetic_data_apriori.csv")   # le dataset
DOSSIER_SORTIE = os.path.join(DOSSIER, "resultats_diabete")            # où écrire les résultats

LS = 0.10                    # support minimum le plus bas (0.10 = 10 % des patients)
BETAS = (0.2, 0.5, 0.8)      # 3 beta croissants : produits faibles, moyens, forts

CONFIANCE_MIN = 0.6          # on garde les règles avec confiance >= 0.6
LIFT_MIN = 0.0               # on garde les règles avec lift >= cette valeur (ex : 1.1)
CONCLUSION_MAX = 1           # nombre max d'items dans la conclusion (0 = sans limite)
CONCLUSION_CONTIENT = ""     # ex : "readmitted" -> seulement les règles dont la conclusion contient ce mot

SEPARATEUR = "  ET  "        # entre deux items (un nom d'item peut contenir " + ", donc pas de "+")
MAX_LIGNES_ECRAN = 15        # nombre de lignes affichées à l'écran (les fichiers .txt contiennent tout)


# =====================================================================
#  1. READ THE DIABETES DATASET
# =====================================================================
def lire_tickets(fichier):
    """Lit le tableau 0/1 du diabète : une ligne = un patient, une colonne = un item.
    Un ticket = la liste des colonnes qui valent 1 (ou True) pour ce patient."""
    tickets = []
    with open(fichier, encoding="utf-8") as f:
        lecteur = csv.reader(f)
        noms = next(lecteur)                           # 1re ligne = nom des colonnes
        for valeurs in lecteur:
            ticket = sorted(noms[i] for i, v in enumerate(valeurs) if v in ("1", "True"))
            if ticket:
                tickets.append(ticket)
    return tickets


# =====================================================================
#  2. RUN CFP-GROWTH
# =====================================================================
def lancer_cfp_growth(tickets):
    """Lance CFP-Growth (MIS par quartiles) et affiche un résumé."""
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
    return res["frequents"]                            # {("item1", "item2"): nombre de patients}


# =====================================================================
#  3. GENERATE ASSOCIATION RULES
# =====================================================================
def preparer_bitsets(tickets):
    """Astuce de rapidité : chaque item devient un grand nombre binaire (1 bit par ticket).
    Le support de (a + b) = nombre de bits à 1 dans (a AND b)."""
    bits = {}
    for i, ticket in enumerate(tickets):
        for item in set(ticket):
            bits[item] = bits.get(item, 0) | (1 << i)
    return bits


def calculer_support(items, bitsets, deja_calcule):
    """Nombre de tickets qui contiennent TOUS les items. Les résultats sont gardés dans deja_calcule."""
    cle = tuple(sorted(items))
    if cle not in deja_calcule:
        nombre = bitsets[cle[0]]
        for item in cle[1:]:
            nombre &= bitsets[item]
        deja_calcule[cle] = nombre.bit_count()
    return deja_calcule[cle]


def generer_regles(frequents, N, bitsets):
    """À partir des itemsets fréquents, fabrique toutes les règles SI ... ALORS ...
    et garde celles qui respectent les réglages (confiance, lift, conclusion).
    Renvoie une liste de dictionnaires (une règle par dictionnaire)."""
    deja_calcule = dict(frequents)                     # on réutilise les supports connus par CFP-Growth
    mot = CONCLUSION_CONTIENT.lower()
    regles = []
    for groupe, sup in frequents.items():
        taille = len(groupe)
        if taille < 2:
            continue                                   # une règle a besoin d'au moins 2 items
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
    regles.sort(key=lambda r: (-r["lift"], -r["confiance_%"]))   # les meilleures règles en premier
    return regles


# =====================================================================
#  4. SAVE RESULTS AS TABLES
#  Une colonne = (clé du dictionnaire, titre affiché, est-ce un nombre ?)
# =====================================================================
COLONNES_ITEMSETS = [("itemset", "Itemset", False), ("taille", "Taille", True),
                     ("nb_patients", "Nb patients", True), ("support_%", "Support (%)", True)]

COLONNES_REGLES = [("si", "SI (conditions)", False), ("alors", "ALORS (conclusion)", False),
                   ("nb_patients", "Nb patients", True), ("support_%", "Support (%)", True),
                   ("confiance_%", "Confiance (%)", True), ("lift", "Lift", True)]


def construire_itemsets(frequents, N):
    """Lignes du tableau des itemsets : triés par taille, puis du plus au moins fréquent."""
    lignes = [{"itemset": SEPARATEUR.join(g), "taille": len(g), "nb_patients": s, "support_%": round(100 * s / N, 2)}
              for g, s in frequents.items()]
    return sorted(lignes, key=lambda l: (l["taille"], -l["nb_patients"]))


def tableau_texte(titre, colonnes, lignes, max_lignes=None, largeur_texte=None):
    """Fabrique un tableau texte bien aligné (textes coupés avec … si largeur_texte est donnée)."""
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
    """Écrit le tableau en .txt (lisible) et en .csv (pour Excel / pandas), puis l'affiche à l'écran."""
    base = os.path.join(DOSSIER_SORTIE, nom)
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(tableau_texte(titre, colonnes, lignes) + "\n")
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:     # utf-8-sig : Excel lit les accents
        ecrivain = csv.writer(f)
        ecrivain.writerow([t for _, t, _ in colonnes])
        ecrivain.writerows([[l[cle] for cle, _, _ in colonnes] for l in lignes])
    print("\n" + tableau_texte(titre, colonnes, lignes, MAX_LIGNES_ECRAN, largeur_texte=48))


# =====================================================================
#  PROGRAMME PRINCIPAL
# =====================================================================
def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    # 1. Lire les données
    tickets = lire_tickets(FICHIER)
    N = len(tickets)
    nb_items = len({p for t in tickets for p in t})
    print("Données : %d patients, %d items, %.1f items par patient en moyenne"
          % (N, nb_items, sum(len(t) for t in tickets) / N))
    print("LS = %g, beta = %s\n" % (LS, BETAS))

    # 2. CFP-Growth
    frequents = lancer_cfp_growth(tickets)

    # 3. Règles d'association
    regles = generer_regles(frequents, N, preparer_bitsets(tickets))

    # 4. Sauvegarde (txt + csv) et aperçu à l'écran
    sauver("regles", "RÈGLES D'ASSOCIATION  (confiance >= %g%%, triées par lift)  -  %d règles"
           % (100 * CONFIANCE_MIN, len(regles)), COLONNES_REGLES, regles)
    sauver("itemsets", "ITEMSETS FRÉQUENTS  -  %d itemsets" % len(frequents),
           COLONNES_ITEMSETS, construire_itemsets(frequents, N))

    print("\nFichiers enregistrés dans :", DOSSIER_SORTIE, "(regles.txt, itemsets.txt, + .csv)")


if __name__ == "__main__":
    main()