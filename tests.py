import numpy as np
import pandas as pd
import h5py

from chrono import start, stop
from constants import ABNORMALITIES, DISEASES, ECG_DIR, PTB_DIR


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


def load_prediction(f_name, diseases):
    raw = np.load(f_name)
    pred = np.round(raw).astype(np.int8)
    pred = pd.DataFrame(pred, columns=diseases)
    add_abnormalities(pred)
    return raw, pred


def remove_abnormalities(columns):
    return list(filter(lambda col: col != "Nb Abnormalities", columns))


def cross_tables(truth, pred):
    def error_percentage(table):
        return (table.loc[0, 1] + table.loc[1, 0]) / table.values.sum()

    tables = dict()
    total = pd.DataFrame(np.zeros((2, 2), dtype=np.int16))
    col = remove_abnormalities(pred.columns)
    for abn in col:
        table = pd.crosstab(truth[abn], pred[abn])
        total += table
        tables[abn] = {"errors": table, "percentage": error_percentage(table)}
    tables["total"] = {"errors": total, "percentage": error_percentage(total)}
    return tables


def make_mmm(datas, pred):
    infos = pd.DataFrame()
    col = remove_abnormalities(pred.columns)
    for abn in col:
        index = pred[pred[abn] == 1].index
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


# %%

if __name__ == "__main__":
    start("tests.py main")

    # X = load_datas(ecg_dir + "ecg_tracings.hdf5")
    # A = pd.read_csv(ecg_dir + "attributes.csv")

    _, pred = load_prediction("ribeiro_output.npy", ABNORMALITIES)
    _, pred_train = load_prediction("ptbxl_train_output.npy", DISEASES)
    _, pred_test = load_prediction("ptbxl_test_output.npy", DISEASES)

    Y = load_annotations(ECG_DIR + "annotations/gold_standard.csv")
    Y_train = load_annotations(PTB_DIR + "annot_train.csv")
    Y_test = load_annotations(PTB_DIR + "annot_test.csv")

    ribeiro_err = cross_tables(Y, pred)
    train_err = cross_tables(Y_train, pred_train)
    test_err = cross_tables(Y_test, pred_test)

    stop()
