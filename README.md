# SWD-Detection-in-Rats
The repository of the *Automatic Detection of the Spike-and-Wave Discharges in Absence Epilepsy for Humans and Rats using Deep Learning* study published in [Biomedical Signal Processing and Control](https://www.sciencedirect.com/science/article/pii/S1746809422002488).


![model](https://user-images.githubusercontent.com/83376963/163556139-0b8fd545-6c5f-4143-aac0-44980bbe1567.png)
## Initial Setup
#### The model files are transferred to the Google Drive due to Git LFS restrictions.
#### One should download the folders in the Google Drive URL below and "replace" with the corresponding folders in the GitHub repository before running the scripts to get meaningful results. 
[Raw and Parsed data](https://drive.google.com/drive/folders/1oIhVsMshzddXUUVGAm8L02yMjcH8NiJq?usp=sharing)
The first section of the EDF_LABELLER.m script can be modified to parse the new EEG records, the existing one parses the shared records based on the seizure occurence information given in the excel sheets. 
The dependencies in [requirements.txt](https://github.com/kutay-ugurlu/SWD-Detection-in-Rats/blob/master/Rat/requirements.txt) should be installed. 

## Running Scripts 
Utility scripts except [create_training_data.py](https://github.com/kutay-ugurlu/SWD-Detection-in-Rats/blob/master/Rat/create_training_data.py) are directly coded in main training scripts.
Training scripts [RatTrainPSD.py](https://github.com/kutay-ugurlu/SWD-Detection-in-Rats/blob/master/Rat/RatTrainPSD.py) and [RatTrainTime.py](https://github.com/kutay-ugurlu/SWD-Detection-in-Rats/blob/master/Rat/RatTrainTime.py) are ready to run and they produce results in JSON and CSV directories. 

## Comparison Study 
The proposed model is compared to more classical approaches in [<ins>SWD Detect with SVM and Tree</ins>](https://github.com/kutay-ugurlu/SWD-Detect-with-SVM-and-Tree)

## Citation
```
@article{NNAbsenceEEG,\
title = {Automatic detection of the spike-and-wave discharges in absence epilepsy for humans and rats using deep learning},\
journal = {Biomedical Signal Processing and Control},\
volume = {76},\
pages = {103726},\
year = {2022},\
issn = {1746-8094},\
doi = {https://doi.org/10.1016/j.bspc.2022.103726},\
url = {https://www.sciencedirect.com/science/article/pii/S1746809422002488}, \
author = {Oguzhan Baser and Melis Yavuz and Kutay Ugurlu and Filiz Onat and Berken Utku Demirel},\
keywords = {Electroencephalography (EEG), Spike-and-wave (SWD), Absence epilepsy, Power spectral density, Deep learning},\
}
```
