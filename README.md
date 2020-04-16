# RPNet

A Deep Learning approach for robust R Peak detection in noisy ECG

## Research

### Architecture Diagram

<p align="center">
  <image src = 'imgs/Unet_5.png' >
</p>

### Datasets

1) Chinese Physiological Signal Challenge (CPSC 2019)  
2) MITBIH Database
3) MIT-BIH ST Change Database
4) Noise Stress Test DataBase (NSTDB) 

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
* Train code will be made available soon

-- To Evaluate the model
* Download the model trained on all of CPSC dataset
* To evaluate on CPSC: `sh evaluate_detectors_CPSC.sh`
* To evaluate on any one of the other three datasets: `sh evaluate_detectors_CPSC.sh`
* To evaluate on the NSTDB dataset: `sh evaluate_nstdb.sh` 

Details on possible changes that can be made to the scipt will be mentioned in the script
