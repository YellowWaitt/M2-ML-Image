import os
import sys
import subprocess

from chrono import chrono
from constants import SCRIPTS_DIR, ECG_DIR, PTB_DIR


def rename(src, dst):
    try:
        os.rename(src, dst)
    except OSError as e:
        print(e, file=sys.stderr)


def call(cmd, name):
    print(cmd)
    try:
        code = subprocess.call(cmd, shell=True)
        if code < 0:
            print("%s was terminated by signal" % name, -code, file=sys.stderr)
        else:
            print("%s returned" % name, code, file=sys.stderr)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)


# %%

@chrono
def train(ecg_hdf5, annot_csv, model_name):
    cmd = "python {0}train.py {1} {2}".format(
        SCRIPTS_DIR, ecg_hdf5, annot_csv)
    call(cmd, "train.py")

    rename("backup_model_best.hdf5", "backup_" + model_name + "_best.hdf5")
    rename("backup_model_last.hdf5", "backup_" + model_name + "_last.hdf5")
    rename("final_model.hdf5", model_name + "_model.hdf5")


@chrono
def predict(ecg_hdf5, model_name, pred_name=None):
    if model_name is None:
        model_hdf5 = "final_model.hdf5"
    else:
        model_hdf5 = model_name + "_model.hdf5"

    cmd = "python {0}predict.py {1} {2}".format(
        SCRIPTS_DIR, ecg_hdf5, model_hdf5)
    call(cmd, "predict.py")

    pred_name = "" if pred_name is None else "_" + pred_name
    rename("dnn_output.npy", model_name + pred_name + "_output.npy")


# %%

def train_ribeiro():
    train(ECG_DIR + "ecg_tracings.hdf5",
          ECG_DIR + "annotations/gold_standard.csv",
          "ribeiro")


def predict_ribeiro():
    predict(ECG_DIR + "ecg_tracings.hdf5", "ribeiro")


# %%

def train_ptbxl():
    train(PTB_DIR + "ecgs_train.hdf5",
          PTB_DIR + "annot_train.csv",
          "ptbxl")


def predict_ptbxl(name):
    predict(PTB_DIR + "ecgs_%s.hdf5" % name, "ptbxl", name)
