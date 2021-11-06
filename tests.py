import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

from sklearn.decomposition import PCA

import time


# %% Chrono decorator

def chrono(fun):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = fun(*args, **kwargs)
        end = time.time()
        print("%s done in %f seconds" % (fun.__name__, end - start))
        return res

    return wrapper


# %% Some constants

LEADS = ["DI", "DII", "DIII", "AVL", "AVF", "AVR",
         "V1", "V2", "V3", "V4", "V5", "V6"]

ABNORMALITIES = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]


# %% Loading functions

@chrono
def load_datas(f_name):
    with h5py.File(f_name, "r") as f:
        datas = np.array(f['tracings'])
    return datas


def add_abnormalities(data_frame):
    data_frame["Nb Abnormalities"] = data_frame.sum(axis=1).astype(np.int8)


def load_annotations(f_name):
    annot = pd.read_csv(f_name, dtype=np.int8)
    add_abnormalities(annot)
    return annot


def load_prediction(f_name):
    raw = np.load(f_name)
    pred = np.round(raw).astype(np.int8)
    pred = pd.DataFrame(pred, columns=ABNORMALITIES)
    add_abnormalities(pred)
    return raw, pred


# %% Plot functions

def subplot_leads(plot_fun, datas, *,
                  title=None,
                  n_row=4, n_col=3,
                  width=4, height=4):
    fig, axs = plt.subplots(n_row, n_col,
                            figsize=(n_col * width, n_row * height))

    for i in range(n_row):
        for j in range(n_col):
            k = i * n_col + j
            ax = axs[i, j]
            plot_fun(datas, k, ax)
            ax.set_title(LEADS[k], size=30)

    if title is not None:
        fig.suptitle(title, size=40, y=1)

    fig.tight_layout()
    plt.show()


def plot_ECG(ecgs, **kwargs):
    def plot(ecgs, k, ax):
        for ecg in ecgs:
            ax.plot(ecg[:, k])

    if isinstance(ecgs, np.ndarray):
        if len(ecgs.shape) == 2:
            ecgs = [ecgs]

    subplot_leads(plot, ecgs, width=8, **kwargs)


def plot_ECG_no(datas, n):
    plot_ECG(datas[n], title=n)


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


# %% Data manipulations

def cross_tables(truth, pred):
    tables = dict()
    total = pd.DataFrame(np.zeros((2, 2), dtype=np.int32))
    for abn in ABNORMALITIES:
        table = pd.crosstab(truth[abn], pred[abn])
        total += table
        tables[abn] = table
    tables["total"] = total
    return tables


def make_mmm(datas, preds):
    infos = pd.DataFrame()
    for abn in ABNORMALITIES:
        index = preds[preds[abn] == 1].index
        infos[abn] = pd.Series({
            "min": np.min(datas[index], axis=0),
            "mean": np.mean(datas[index], axis=0),
            "max": np.max(datas[index], axis=0),
        })
    infos["total"] = pd.Series({
        "min": np.min(datas, axis=0),
        "mean": np.mean(datas, axis=0),
        "max": np.max(datas, axis=0),
    })
    return infos


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


# %% Classifications tests


# %% Main

if __name__ == '__main__':
    X = load_datas("data/ecg_tracings.hdf5")
    Y = load_annotations("data/annotations/gold_standard.csv")
    A = pd.read_csv("data/attributes.csv")

    _, pred = load_prediction("dnn_output.npy")
    ct = cross_tables(Y, pred)

    pcas, pca_stacked, pca_flat, projs, stacked, X_flat = make_PCA(X)
