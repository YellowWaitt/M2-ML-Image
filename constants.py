from pathlib import Path

# %% Directory's paths

ROOT_DIR = Path(__file__).parent.as_posix() + "/"
SCRIPTS_DIR = ROOT_DIR + "automatic-ecg-diagnosis-master/"
DATA_DIR = ROOT_DIR + "data/"

ECG_DIR = DATA_DIR + "ecg/"
PTB_DIR = DATA_DIR + "ptb-xl/"

# %% Diseases tags

LEADS = ["DI", "DII", "DIII", "AVL", "AVF", "AVR",
         "V1", "V2", "V3", "V4", "V5", "V6"]

ABNORMALITIES = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
DISEASES = ["MI", "HYP", "CD", "STTC"]
