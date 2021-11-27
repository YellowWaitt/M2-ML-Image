import os
import sys
import subprocess

from chrono import chrono
from path import scripts_dir, ecg_dir, ptb_dir


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
        scripts_dir, ecg_hdf5, annot_csv)
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
        scripts_dir, ecg_hdf5, model_hdf5)
    call(cmd, "predict.py")

    pred_name = "" if pred_name is None else "_" + pred_name
    rename("dnn_output.npy", model_name + pred_name + "_output.npy")


# %%

def train_ribeiro():
    train(ecg_dir + "ecg_tracings.hdf5",
          ecg_dir + "annotations/gold_standard.csv",
          "ribeiro")


def predict_ribeiro():
    predict(ecg_dir + "ecg_tracings.hdf5", "ribeiro")


# %%

def train_ptbxl():
    train(ptb_dir + "ecgs_train.hdf5",
          ptb_dir + "annot_train.csv",
          "ptbxl")


def predict_ptbxl(name):
    predict(ptb_dir + "ecgs_%s.hdf5" % name, "ptbxl", name)
