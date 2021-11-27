import pandas as pd
import numpy as np
import h5py

import ast
import cv2
from tqdm import tqdm
import wfdb

from chrono import chrono
from path import ptb_dir
from tests import add_abnormalities


DISEASES = ["MI", "HYP", "CD", "STTC"]


@chrono
def load_ptbxl(*, size=32):
    def delete_unk_ecg(X, Y):
        Y = list(Y)
        X_, Y_ = [], []
        for i, dis in enumerate(Y):
            if len(dis) > 0:
                X_.append(X[i])
                Y_.append(Y[i])
        X_ = np.array(X_, dtype="float%d" % size)
        Y_ = np.array(Y_, dtype=object)
        return X_, Y_

    def Ribeiro_convert(Y):
        Y_ = np.zeros((Y.shape[0], 4), dtype=np.int8)
        for i, info in enumerate(Y):
            for disease in info:
                if disease != "NORM":
                    Y_[i, DISEASES.index(disease)] = 1
        Y_ = pd.DataFrame(Y_, columns=DISEASES)
        return Y_

    def resize(X, scale):
        X_conv = np.empty((X.shape[0], scale, 12), dtype="float%d" % size)
        for i, array in enumerate(X):
            X_conv[i, :, :] = cv2.resize(array, (12, scale))
        return X_conv

    def process(X, Y):
        Y = Y.diagnostic_superclass
        X, Y = delete_unk_ecg(X, Y)
        Y = Ribeiro_convert(Y)
        X = resize(X, 4096)
        return X, Y

    X_train, Y_train, X_test, Y_test = load_raw(ptb_dir, size=size)
    X_train, Y_train = process(X_train, Y_train)
    X_test, Y_test = process(X_test, Y_test)
    return X_train, Y_train, X_test, Y_test


@chrono
def load_raw(path, *, size=32, sampling_rate=500, test_fold=10):
    @chrono
    def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            iter = df.filename_lr
        else:
            iter = df.filename_hr
        data = [wfdb.rdsamp(path+f, return_res=size) for f in tqdm(iter)]
        data = np.array([signal for signal, meta in data],
                        dtype="float%d" % size)
        return data

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # load and convert annotation data
    Y = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y["scp_codes"].apply(lambda x: ast.literal_eval(x))
    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df["diagnostic"] == 1]
    # Apply diagnostic superclass
    Y["diagnostic_superclass"] = Y["scp_codes"].apply(aggregate_diagnostic)
    # Split data into train and test
    # Train
    index = Y["strat_fold"] != test_fold
    X_train = X[np.where(index)]
    Y_train = Y[index]
    # Test
    index = ~index
    X_test = X[np.where(index)]
    Y_test = Y[index]

    return X_train, Y_train, X_test, Y_test


@chrono
def convert_for_model(*args):
    def save_hdf5(X, name):
        with h5py.File(name, "w") as f:
            f.create_dataset("tracings", data=X)

    def save_csv(Y, name):
        Y.to_csv(name, index=False)

    def save(X, Y, name):
        save_hdf5(X, ptb_dir + "ecgs_%s.hdf5" % name)
        save_csv(Y, ptb_dir + "annot_%s.csv" % name)

    if len(args) == 0:
        X_train, Y_train, X_test, Y_test = load_ptbxl()
    else:
        X_train, Y_train, X_test, Y_test = args

    save(X_train, Y_train, "train")
    save(X_test, Y_test, "test")


@chrono
def load_ptbxl2():
    def load_hdf5(name):
        with h5py.File(name, "r") as f:
            X = np.array(f["tracings"])
        return X

    def load_csv(name):
        Y = pd.read_csv(name, dtype=np.int8)
        add_abnormalities(Y)
        return Y

    def load(name):
        X = load_hdf5(ptb_dir + "ecgs_%s.hdf5" % name)
        Y = load_csv(ptb_dir + "annot_%s.csv" % name)
        return X, Y

    X_train, Y_train = load("train")
    X_test, Y_test = load("test")
    return X_train, Y_train, X_test, Y_test


# %%

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_ptbxl2()
