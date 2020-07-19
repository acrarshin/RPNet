# RPNet

A Deep Learning approach for robust R Peak detection in noisy ECG

## Research

### Architecture Diagram

<p align="center">
  <image src = 'imgs/Unet_5.png' >
</p>

### Datasets

1) [Chinese Physiological Signal Challenge (CPSC 2019)](http://2019.cpscsub.com/)
2) [MITBIH Database](https://www.physionet.org/content/mitdb/1.0.0/)
3) [MIT-BIH ST Change Database](http://physionet.incor.usp.br/physiobank/database/stdb/)
4) [Noise Stress Test DataBase (NSTDB)](https://www.physionet.org/content/nstdb/1.0.0/) 

### Quantitative Comparisons
Evaluation of model and three traditional methods on CPSC dataset

<p align="center">
  <image src = 'imgs/CPSC_eval.png' >
</p>

Evalation of Model on the other 3 datasets

<p align="center">
  <image src = 'imgs/Perf_on_3_datasets.png' >
</p>

Evaluation of Model in presence of noise(SNR wise) on NSTDB

<p align="center">
  <image src = 'imgs/Perf_on_NSTDB.png' >
</p>

### Qualitative Results
<p align="center">
  <image src = 'imgs/Collage_results.png' >
</p>

### Steps 

-- General Steps
* Download all the 4 datasets.

-- To train the model
* Run train_CPSC.ipynb

-- To Evaluate the model
* [Download the model](https://drive.google.com/file/d/19xN7pZsALb09bxWjrSKdAlJmRqYL0M0g/view?usp=sharing)
* To evaluate on CPSC: `sh evaluate_detectors_CPSC.sh`
* To evaluate on any one of the other three datasets: `sh evaluate_detectors_CPSC.sh`
* To evaluate on the NSTDB dataset: `sh evaluate_nstdb.sh` 

Details on possible changes that can be made to the scipt will be mentioned in the script. We would like to acknowledge Mr Bernd Porr whose repo we forked for the [implementation](https://github.com/berndporr/py-ecg-detectors) of the traditional ECG Detectors.
