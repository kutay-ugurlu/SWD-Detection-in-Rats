# SWD-Detection-in-Rats
The repository of the *Automatic Detection of the Spike-and-Wave Discharges in Absence Epilepsy for Humans and Rats using Deep Learning* study accepted for publication in Biomedical Signal Processing and Control.
## Initial Setup
#### The model files are transferred to the Google Drive due to Git LFS restrictions.
#### One should download the folders in the Google Drive URL below and "replace" with the corresponding folders in the GitHub repository before running the scripts to get meaningful results. 
[Raw and Parsed data](https://drive.google.com/drive/folders/1oIhVsMshzddXUUVGAm8L02yMjcH8NiJq?usp=sharing)
The first section of the EDF_LABELLER.m script can be modified to parse the new EEG records, the existing one parses the shared records based on the seizure occurence information given in the excel sheets. 
The dependencies in [requirements.txt]((https://github.com/kutay-ugurlu/SWD-Detection-in-Rats/blob/master/Rat/requirements.txt) should be installed. 

## Running Scripts 
Utility scripts except [create_training_data.py](https://github.com/kutay-ugurlu/SWD-Detection-in-Rats/blob/master/Rat/create_training_data.py) are directly coded in main training scripts.
Training scripts [RatTrainPSD.py](https://github.com/kutay-ugurlu/SWD-Detection-in-Rats/blob/master/Rat/RatTrainPSD.py) and [RatTrainTime.py](https://github.com/kutay-ugurlu/SWD-Detection-in-Rats/blob/master/Rat/RatTrainTime.py) are ready to run and they produce results in JSON and CSV directories. 

## Comparison Study 
The proposed model is compared to more classical approaches in [SWD Detect with SVM and Tree](https://github.com/kutay-ugurlu/SWD-Detect-with-SVM-and-Tree)
