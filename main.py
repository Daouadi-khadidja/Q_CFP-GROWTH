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
#     4. Save CSV results              -> ecrire_csv
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
DOSSIER_SORTIE = os.path.join(DOSSIER, "resultats_diabete")            # où écrire les CSV

LS = 0.10                    # support minimum le plus bas (0.10 = 10 % des patients)
BETAS = (0.2, 0.5, 0.8)      # 3 beta croissants : produits faibles, moyens, forts

CONFIANCE_MIN = 0.6          # on garde les règles avec confiance >= 0.6
LIFT_MIN = 0.0               # on garde les règles avec lift >= cette valeur (ex : 1.1)
CONCLUSION_MAX = 1           # nombre max d'items dans la conclusion (0 = sans limite)
DEBUT_CONCLUSION = ""        # ex : "readmitted" -> seulement les règles qui prédisent readmitted=...


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
    regles = []
    for groupe, sup in frequents.items():
        taille = len(groupe)
        if taille < 2:
            continue                                   # une règle a besoin d'au moins 2 items
        limite = taille - 1 if CONCLUSION_MAX == 0 else min(CONCLUSION_MAX, taille - 1)
        for k in range(1, limite + 1):
            for conclusion in combinations(groupe, k):
                if DEBUT_CONCLUSION and not all(c.startswith(DEBUT_CONCLUSION) for c in conclusion):
                    continue
                gauche = [i for i in groupe if i not in conclusion]
                confiance = sup / calculer_support(gauche, bitsets, deja_calcule)
                if confiance < CONFIANCE_MIN:
                    continue
                lift = confiance / (calculer_support(conclusion, bitsets, deja_calcule) / N)
                if lift < LIFT_MIN:
                    continue
                regles.append({"si": " + ".join(gauche), "alors": " + ".join(conclusion),
                               "nb_patients": sup, "support_%": round(100 * sup / N, 2),
                               "confiance": round(confiance, 3), "lift": round(lift, 3)})
    regles.sort(key=lambda r: (-r["lift"], -r["confiance"]))   # les meilleures règles en premier
    return regles


# =====================================================================
#  4. SAVE CSV RESULTS
# =====================================================================
def ecrire_csv(chemin, colonnes, lignes):
    """Enregistre une liste de dictionnaires dans un fichier CSV."""
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        ecrivain = csv.DictWriter(f, fieldnames=colonnes)
        ecrivain.writeheader()
        ecrivain.writerows(lignes)


def sauver_itemsets(frequents, N):
    """itemsets.csv : triés par taille, puis du plus au moins fréquent."""
    lignes = [{"itemset": " + ".join(g), "taille": len(g), "nb_patients": s, "support_%": round(100 * s / N, 2)}
              for g, s in frequents.items()]
    lignes.sort(key=lambda l: (l["taille"], -l["nb_patients"]))
    ecrire_csv(os.path.join(DOSSIER_SORTIE, "itemsets.csv"), ["itemset", "taille", "nb_patients", "support_%"], lignes)


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

    # 4. Sauvegarde en CSV
    sauver_itemsets(frequents, N)
    ecrire_csv(os.path.join(DOSSIER_SORTIE, "regles.csv"),
               ["si", "alors", "nb_patients", "support_%", "confiance", "lift"], regles)

    print("\nRègles (confiance >= %g) : %d" % (CONFIANCE_MIN, len(regles)))
    for r in regles[:10]:
        print("SI %-40s ALORS %-25s support %5.1f %% | confiance %.2f | lift %.2f"
              % (r["si"], r["alors"], r["support_%"], r["confiance"], r["lift"]))
    print("\nFichiers enregistrés dans :", DOSSIER_SORTIE)


if __name__ == "__main__":
    main()