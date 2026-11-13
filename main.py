# Lance tout : lecture des données, MIS, CFP-Growth, règles, sauvegarde (python main.py).
import csv
import os
import time

from cfp_growth import TropDeMotifs, cfp_growth, lire_taille_max
from mis import calculer_mis_categories, ecrire_mis_txt
from regles import generer_regles, preparer_bitsets

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# Réglages.
FICHIER = os.path.join(DOSSIER, "Data", "Lim_solar_data_net.csv")
DOSSIER_SORTIE = os.path.join(DOSSIER, "resultats_solaire")
FICHIER_MIS = os.path.join(DOSSIER_SORTIE, "mis.txt")

TAILLE_MAX = 4
LIMITE_MOTIFS = 3_000_000

CONFIANCE_MIN = 0.6
LIFT_MIN = 1.0
CONCLUSION_MAX = 0
CONCLUSION_CONTIENT = ""

SEPARATEUR = "  ET  "
MAX_LIGNES_ECRAN = 15

COLONNES_ITEMSETS = [("itemset", "Itemset", False), ("taille", "Taille", True),
                     ("nb_lignes", "Nb lignes", True), ("support_%", "Support (%)", True)]

COLONNES_REGLES = [("si", "SI (conditions)", False), ("alors", "ALORS (conclusion)", False),
                   ("support", "Support relatif", True), ("confiance", "Confiance", True),
                   ("lift", "Lift", True)]


# Enlève les guillemets et antislashs de l'export Weka.
def nettoyer(valeur):
    return valeur.replace("\\", "").replace("'", "").strip()


# Lit le fichier : une ligne = un ticket = une liste d'items.
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


# Lance CFP-Growth et affiche un résumé.
def lancer_cfp_growth(tickets, m):
    debut = time.perf_counter()
    try:
        res = cfp_growth(tickets, FICHIER_MIS, taille_max=TAILLE_MAX, limite_motifs=LIMITE_MOTIFS)
    except TropDeMotifs as e:
        print("ARRÊT : trop d'itemsets (%s). Baisse TAILLE_MAX." % e)
        raise SystemExit(1)
    duree = time.perf_counter() - debut

    N = len(tickets)
    q1, q3 = m["coupures"]
    print("Quartiles : Q1 = %.0f lignes (%.1f %%), Q3 = %.0f lignes (%.1f %%)" % (q1, 100 * q1 / N, q3, 100 * q3 / N))
    bf, bm, bF = m["betas"]
    print("Bêtas : faible = %.5f, moyen = %.5f, fort = %.5f" % (bf, bm, bF))
    print("Seuils MIS (tickets) : min = %d, max = %d" % (min(m["seuils"].values()), max(m["seuils"].values())))
    print("Taille max des itemsets : %s" % (lire_taille_max(TAILLE_MAX) or "pas de limite"))
    print("Itemsets fréquents : %d (%.1f s)" % (len(res["frequents"]), duree))
    return res["frequents"]


# Trie les itemsets par taille puis par fréquence.
def construire_itemsets(frequents, N):
    lignes = [{"itemset": SEPARATEUR.join(g), "taille": len(g), "nb_lignes": s, "support_%": round(100 * s / N, 2)}
              for g, s in frequents.items()]
    return sorted(lignes, key=lambda l: (l["taille"], -l["nb_lignes"]))


# Fabrique un tableau texte aligné.
def tableau_texte(titre, colonnes, lignes, max_lignes=None, largeur_texte=None):
    def case(valeur, nombre):
        valeur = str(valeur)
        return valeur if nombre or not largeur_texte or len(valeur) <= largeur_texte else valeur[:largeur_texte - 1] + "…"

    cases = [[case(l[cle], nombre) for cle, _, nombre in colonnes] for l in lignes[:max_lignes]]
    largeurs = [max([len(t)] + [len(c[j]) for c in cases]) for j, (_, t, _) in enumerate(colonnes)]

    def ligne(valeurs):
        return "| " + " | ".join(v.rjust(w) if colonnes[j][2] else v.ljust(w)
                                 for j, (v, w) in enumerate(zip(valeurs, largeurs))) + " |"

    trait = "+-" + "-+-".join("-" * w for w in largeurs) + "-+"
    sortie = [titre, trait, ligne([t for _, t, _ in colonnes]), trait] + [ligne(c) for c in cases] + [trait]
    if max_lignes and len(lignes) > max_lignes:
        sortie.append("... %d lignes de plus dans les fichiers .txt / .csv" % (len(lignes) - max_lignes))
    return "\n".join(sortie)


# Écrit le tableau en .txt et .csv puis l'affiche.
def sauver(nom, titre, colonnes, lignes):
    base = os.path.join(DOSSIER_SORTIE, nom)
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(tableau_texte(titre, colonnes, lignes) + "\n")
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow([t for _, t, _ in colonnes])
        ecrivain.writerows([[l[cle] for cle, _, _ in colonnes] for l in lignes])
    print("\n" + tableau_texte(titre, colonnes, lignes, MAX_LIGNES_ECRAN, largeur_texte=48))


# Programme principal.
def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    if not os.path.exists(FICHIER):
        print("Fichier introuvable :", FICHIER)
        return

    tickets = lire_tickets(FICHIER)
    N = len(tickets)
    print("Données : %d lignes, %d items" % (N, len({p for t in tickets for p in t})))

    m = calculer_mis_categories(tickets)
    ecrire_mis_txt(m, FICHIER_MIS)
    print("MIS écrits dans :", FICHIER_MIS, "\n")

    frequents = lancer_cfp_growth(tickets, m)

    regles = generer_regles(frequents, N, preparer_bitsets(tickets),
                            confiance_min=CONFIANCE_MIN, lift_min=LIFT_MIN,
                            conclusion_max=CONCLUSION_MAX, conclusion_contient=CONCLUSION_CONTIENT,
                            separateur=SEPARATEUR)

    sauver("regles", "RÈGLES D'ASSOCIATION  (confiance >= %g%%, triées par lift)  -  %d règles"
           % (100 * CONFIANCE_MIN, len(regles)), COLONNES_REGLES, regles)
    sauver("itemsets", "ITEMSETS FRÉQUENTS  -  %d itemsets" % len(frequents),
           COLONNES_ITEMSETS, construire_itemsets(frequents, N))

    print("\nFichiers enregistrés dans :", DOSSIER_SORTIE)


if __name__ == "__main__":
    main()