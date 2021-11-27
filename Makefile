SCRIPTS = ./automatic-ecg-diagnosis-master/
HDF5    = ./data/ecg/ecg_tracings.hdf5
CSV     = ./data/ecg/annotations/gold_standard.csv
MODEL   = ./final_model.hdf5

.PHONY: all train predict model

train:
	python $(SCRIPTS)train.py $(HDF5) $(CSV)

predict:
	python $(SCRIPTS)predict.py $(HDF5) $(MODEL)

model:
	python $(SCRIPTS)model.py > model.txt
