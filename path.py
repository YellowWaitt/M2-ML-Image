from pathlib import Path


root_dir = Path(__file__).parent.as_posix() + "/"
scripts_dir = root_dir + "automatic-ecg-diagnosis-master/"
data_dir = root_dir + "data/"

ecg_dir = data_dir + "ecg/"
ptb_dir = data_dir + "ptb-xl/"
