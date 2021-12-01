import numpy as np
import matplotlib.pyplot as plt

from constants import LEADS


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
