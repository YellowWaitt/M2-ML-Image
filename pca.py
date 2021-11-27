import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from tests import load_datas, subplot_leads, ABNORMALITIES
from chrono import chrono


# %% PCA (sert à rien)

def plot_pca(datas, labels, ax=None):
    COLORS = ["b", "g", "r", "c", "m", "y"]

    if ax is None:
        _, ax = plt.subplots()

    index = labels[labels["Nb Abnormalities"] == 0].index
    to_plot = datas[index]
    ax.scatter(to_plot[:, 0], to_plot[:, 1], c="k", alpha=0.5)

    for c, abn in zip(COLORS, ABNORMALITIES):
        index = labels[labels[abn] == 1].index
        to_plot = datas[index]
        ax.scatter(to_plot[:, 0], to_plot[:, 1], c=c, alpha=0.7)

    return ax


@chrono
def plot_pcas(datas, labels, **kwargs):
    def plot(datas, k, ax):
        plot_pca(datas[k], labels, ax)

    subplot_leads(plot, datas, **kwargs)


@chrono
def make_PCA(datas):
    def ind90(ratio):
        """
        Cherche le nombre de composante qui explique 90% de variance
        """
        cum = 0.0
        for i, r in enumerate(ratio):
            cum += r
            if cum >= 0.9:
                return i + 1

    def pca90(datas):
        """
        Détermine un "bon" nombre de composante et réeffectue une PCA pour
        utiliser les transform et inverse_transform déjà implémentés
        """
        pca = PCA().fit(datas)
        return PCA(ind90(pca.explained_variance_ratio_)).fit(datas)

    pcas = [pca90(datas[:, :, i]) for i in range(datas.shape[2])]

    projs = [pca.transform(datas[:, :, i]) for i, pca in enumerate(pcas)]
    stacked = np.hstack(projs)
    pca_stacked = pca90(stacked)

    flatted = datas.reshape((datas.shape[0], -1))
    pca_flat = pca90(flatted)

    return pcas, pca_stacked, pca_flat, projs, stacked, flatted


# %% Main

if __name__ == "__main__":
    X = load_datas("data/ecg_tracings.hdf5")

    pcas, pca_stacked, pca_flat, projs, stacked, X_flat = make_PCA(X)
