#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import pandas as pd

from utils import path


def influence(feature, year, month=0):
    """Génère le graphe et sauvegarde dans`figures`."""
    figure, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("temps (jours)")

    for dtype, name, axis, color in zip(('conso', 'pertes'), (feature, 'perte'), (ax1, ax2), "br"):
        dataframe = pd.read_csv(path(f"data/{dtype}/{dtype}_{year}.csv"))
        if month:
            dataframe = dataframe.loc[dataframe['mois'] == month]
        data = dataframe.groupby('jour').sum()[name]
        axis.plot(data, f"{color}-")
        axis.set_ylabel(name, color=color)
        axis.tick_params(axis='y', labelcolor=color)
    figure.tight_layout()
    plt.title(f"{feature} {year}/{month}")
    plt.savefig(path(f"figures/{feature}_{year}_{month}.png"))


def compare(reality, prediction, name=None):
    """Compare reality and prediction, save with given file name
    (path and extension auto built).
    """
    plt.figure()
    real, pred = [d.reshape(-1, 24).sum(1) for d in (reality, prediction)]
    plt.plot(real, 'g-', label='réalité')
    plt.plot(pred, 'r-', label="prédiction")
    plt.legend()
    if name:
        plt.savefig(path(f"figures/{name}.png"))
    plt.show()


def main():
    """Pour générer facilement en ligne de commande."""
    parser = argparse.ArgumentParser()
    parser.add_argument('feature', nargs='?', default='consommation')
    parser.add_argument('year', nargs='?', default=2018)
    parser.add_argument('month', nargs='?', default=0)
    args = parser.parse_args()
    influence(args.feature, args.year, args.month)


if __name__ == "__main__":
    main()
