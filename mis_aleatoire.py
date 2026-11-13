# mis_aleatoire.py : fabrique un fichier MIS ALÉATOIRE (aucune formule : chaque MIS est tiré au hasard).
# Il garde les mêmes produits et le même format que mis.txt, pour que cfp_growth.py puisse le lire.
# Lancer : python mis_aleatoire.py   (comparaison_mis.py utilise aussi la fonction ecrire_mis_aleatoire)
import math
import os
import random

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# RÉGLAGES : les seules valeurs à modifier avant de relancer.
FICHIER_MIS = os.path.join(DOSSIER, "resultats_solaire", "mis.txt")              # fichier modèle (donne la liste des produits)
FICHIER_SORTIE = os.path.join(DOSSIER, "resultats_solaire", "mis_aleatoire.txt")  # fichier créé

MIS_MIN = 0.009563    # = LS
MIS_MAX = 0.6    
GRAINE = 42         # même nombre = mêmes valeurs à chaque lancement ; change-le pour obtenir d'autres valeurs aléatoires

SEP = "\t"


# 1. Lit le fichier modèle : le nombre de tickets N (dans les commentaires) et la liste (produit, support).
def lire_modele(chemin):
    N = None
    produits = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.rstrip("\n")
            if ligne.startswith("# N ="):
                N = int(ligne.split("=")[1].split()[0])
            if ligne.strip() == "" or ligne.startswith("#"):
                continue
            parties = ligne.split(SEP)
            produits.append((parties[0], float(parties[1])))      # (produit, support en proportion)
    return N, produits


# 2. Tire un MIS au hasard pour chaque produit et écrit le fichier.
def ecrire_mis_aleatoire(produits, N, chemin, mis_min=MIS_MIN, mis_max=MIS_MAX, graine=GRAINE):
    """produits = liste de (produit, support en proportion), dans l'ordre du fichier mis.txt.
    Renvoie le nombre de produits écrits."""
    random.seed(graine)

    lignes = []
    for produit, support in produits:
        mis = random.uniform(mis_min, mis_max)          # <- le hasard : aucune formule
        seuil = math.ceil(mis * N)                      # nombre minimum de tickets
        lignes.append((produit, support, mis, seuil))
    lignes.sort(key=lambda l: (-l[2], l[0]))

    with open(chemin, "w", encoding="utf-8") as f:
        f.write("# MIS ALÉATOIRES (tirés au hasard, aucune formule)\n")
        f.write("# N = %d tickets\n" % N)
        f.write("# MIS tiré uniformément entre %.2f et %.2f (graine = %d)\n" % (mis_min, mis_max, graine))
        f.write("# produit" + SEP + SEP.join(["support_rel", "categorie", "beta", "MIS_rel", "seuil_tickets"]) + "\n")
        for produit, support, mis, seuil in lignes:
            f.write(SEP.join([produit, "%.9f" % support, "aleatoire", "-", "%.9f" % mis, str(seuil)]) + "\n")
    return len(lignes)


def main():
    N, produits = lire_modele(FICHIER_MIS)
    n = ecrire_mis_aleatoire(produits, N, FICHIER_SORTIE)
    print("%d produits, MIS aléatoires entre %.0f %% et %.0f %%" % (n, 100 * MIS_MIN, 100 * MIS_MAX))
    print("Fichier écrit :", FICHIER_SORTIE)


if __name__ == "__main__":
    main()
