# Nettoie le dataset une seule fois : enlève les guillemets Weka, la colonne Time et les colonnes presque constantes (python nettoyer.py).
import csv
import os
from collections import Counter

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# RÉGLAGES : les seules valeurs à modifier avant de relancer.
FICHIER_ENTREE = os.path.join(DOSSIER, "Data", "Lim_solar_data_cleaned.csv")
FICHIER_SORTIE = os.path.join(DOSSIER, "Data", "Lim_solar_data_net.csv")

COLONNES_A_RETIRER = ["Time"]      # colonnes qui ne sont pas des mesures (index de temps)
SEUIL_CONSTANT = None             # None = on garde les colonnes presque constantes ; 0.80 = on retire celles dont une tranche couvre plus de 80 % des lignes


def nettoyer(valeur):
    """Enlève les guillemets et antislashs de l'export Weka : '\\'(1.2-3.4]\\'' devient (1.2-3.4]."""
    return valeur.replace("\\", "").replace("'", "").strip()


def main():
    with open(FICHIER_ENTREE, encoding="utf-8-sig") as f:
        lecteur = csv.reader(f)
        noms = [nettoyer(n) for n in next(lecteur)]
        lignes = [[nettoyer(v) for v in ligne] for ligne in lecteur]
    N = len(lignes)
    print("Lu : %d lignes, %d colonnes" % (N, len(noms)))

    retirees = {}
    for j, nom in enumerate(noms):
        if nom in COLONNES_A_RETIRER:
            retirees[nom] = "colonne demandée"
        elif SEUIL_CONSTANT is not None and nom != "label":
            tranche, nombre = Counter(l[j] for l in lignes).most_common(1)[0]
            if nombre / N > SEUIL_CONSTANT:
                retirees[nom] = "presque constante (%.1f %% dans %s)" % (100 * nombre / N, tranche)

    gardees = [j for j, nom in enumerate(noms) if nom not in retirees]
    for nom, raison in retirees.items():
        print("   retirée : %-6s %s" % (nom, raison))

    with open(FICHIER_SORTIE, "w", newline="", encoding="utf-8") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow([noms[j] for j in gardees])
        ecrivain.writerows([[l[j] for j in gardees] for l in lignes])
    print("Écrit : %s (%d lignes, %d colonnes : %s)"
          % (FICHIER_SORTIE, N, len(gardees), ", ".join(noms[j] for j in gardees)))


if __name__ == "__main__":
    main()
