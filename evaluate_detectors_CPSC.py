### Importing basic libraries
import scipy.io
import os
import numpy as np
from scipy import signal
import random
import pandas as pd
import scipy
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
import math
import wfdb as wf
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn
import scipy.signal
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as n
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from py_ecg.ecgdetectors import Detectors
import py_ecg._tester_utils
from utils import score,load_model_CNN,obtain_data

def load_patient_data(PATH,PATH_ref,fs_):
    R_ref = []
    HR_ref = []
    
    ## Collecting patient files and associated annotation
    patient_info = []
    patient_reference = []
    for files in sorted(os.listdir(PATH)):
        patient_info.append(scipy.io.loadmat(os.path.join(PATH,files)))
    for files in sorted(os.listdir(PATH_ref)):
        patient_reference.append(scipy.io.loadmat(os.path.join(PATH_ref,files)))
    
    ## Obtaining grouth truth Rpeak annotation and HR values
    for i in range(len(patient_reference)):
        R_ref.append(patient_reference[i]['R_peak'].flatten())
        R_ref[i] = R_ref[i][(R_ref[i] >= 0.5*fs_) & (R_ref[i] <= 9.5*fs_)]
        r_hr = np.array([loc for loc in R_ref[i] if (loc > 5.5 * fs_ and loc < 5000 - 0.5 * fs_)])
        hr = round( 60 * fs_ / np.mean(np.diff(r_hr)))
        HR_ref.append(hr)
    
    ## Obtaining raw ECG record
    patient_ecg = []
    for i in range(0,len(patient_info)):
        patient_ecg.append(patient_info[i]['ecg'])
    patient_ecg = np.asarray(patient_ecg)[:,:,0]
    
    return patient_ecg,patient_reference,R_ref,HR_ref

def score(r_ref, hr_ref, r_ans, hr_ans, fs_, thr_):
    HR_score = 0
    record_flags = np.ones(len(r_ref))
    failed_record = []
    fp = []
    fn = []
    zer = []
    all_TP = 0
    all_FN = 0
    all_FP = 0
    for i in range(len(r_ref)):
        FN = 0
        FP = 0
        TP = 0
        if math.isnan(hr_ans[i]):
            hr_ans[i] = 0
        hr_der = abs(int(hr_ans[i]) - int(hr_ref[i]))
        if hr_der <= 0.02 * hr_ref[i]:
            HR_score = HR_score + 1
        elif hr_der <= 0.05 * hr_ref[i]:
            HR_score = HR_score + 0.75
        elif hr_der <= 0.1 * hr_ref[i]:
            HR_score = HR_score + 0.5
        elif hr_der <= 0.2 * hr_ref[i]:
            HR_score = HR_score + 0.25

        for j in range(len(r_ref[i])):
            loc = np.where(np.abs(r_ans[i] - r_ref[i][j]) <= thr_*fs_)[0]
            if j == 0:
                err = np.where((r_ans[i] >= 0.5*fs_ + thr_*fs_) & (r_ans[i] <= r_ref[i][j] - thr_*fs_))[0]
            elif j == len(r_ref[i])-1:
                err = np.where((r_ans[i] >= r_ref[i][j]+thr_*fs_) & (r_ans[i] <= 9.5*fs_ - thr_*fs_))[0]
            else:
                err = np.where((r_ans[i] >= r_ref[i][j]+thr_*fs_) & (r_ans[i] <= r_ref[i][j+1]-thr_*fs_))[0]

            FP = FP + len(err)
            if len(loc) >= 1:
                TP += 1
                FP = FP + len(loc) - 1
            elif len(loc) == 0:
                FN += 1

        if FN + FP > 1:
            record_flags[i] = 0
        elif FN == 1 and FP == 0:
            record_flags[i] = 0.3
        elif FN == 0 and FP == 1:
            record_flags[i] = 0.7
        ### Custom
        if(record_flags[i] != 1):
            failed_record.append(i)
        if(record_flags[i] == 0.7):
            fp.append(i)
        elif(record_flags[i] == 0.3):
            fn.append(i)
        elif(record_flags[i] == 0):
            zer.append(i)
        
        
        all_FP += FP
        all_FN += FN
        all_TP += TP
        
    Recall = all_TP / (all_FN + all_TP)
    Precision = all_TP / (all_FP + all_TP)
    F1_score = 2 * Recall * Precision / (Recall + Precision)
    print("TP's:{} FN's:{} FP's:{}".format(all_TP,all_FN,all_FP))
    print("Recall:{}, Precision(FNR):{}, F1-Score:{}".format(Recall,Precision,F1_score))
        ###
    ##Custom
    print("Failed_records:",failed_record)
    print("FP's:",fp)
    print("FN's:",fn)
    print("Zeros's:",zer)
    ##    
    rec_acc = round(np.sum(record_flags) / len(r_ref), 4)
    hr_acc = round(HR_score / len(r_ref), 4)

    return rec_acc, hr_acc


def call_detector(args,patient_ecg,patient_ref,R_ref,fs):
    windowed_data = patient_ecg[1936:,:]
    windowed_beats  = patient_ref[1936:]
    r_peaks_ans = []
    count = 0
    detectors = Detectors(fs)
    peak_locs = []

    tolerance = int(fs * 0.075)
    max_delay_in_samples = 500 / 5 
    i = 0
    results = np.zeros((len(patient_ecg), 5), dtype=int)
    for records in tqdm(range(len(windowed_data))):     
        if (args.algorithm == 0):
            r_peaks = detectors.swt_detector(windowed_data[records]) 
        elif (args.algorithm == 1):
            r_peaks = detectors.hamilton_detector(windowed_data[records]) 
        elif (args.algorithm == 2):
            r_peaks = detectors.christov_detector(windowed_data[records]) 
        peak_locs.append(r_peaks)
        anno = windowed_beats[records]
        delay = py_ecg._tester_utils.calcMedianDelay(r_peaks, windowed_data[records], max_delay_in_samples)
        if delay > 1:
            TP, FP, FN = py_ecg._tester_utils.evaluate_detector(r_peaks, anno, delay, tol=tolerance)
            TN = len(windowed_data[records])-(TP+FP+FN)
            results[i, 0] = int(records)    
            results[i, 1] = TP
            results[i, 2] = FP
            results[i, 3] = FN
            results[i, 4] = TN
            i += 1
                
    return peak_locs,results
    
def compute_HR(peak_locs,fs_):
    HR_ans = []
    peak_locs_1 = []
    for peaks in peak_locs:
        r_hr = np.array([loc for loc in peaks if (loc > 5.5 * fs_ and loc < 5000 - 0.5 * fs_)])
        hr = round( 60 * fs_ / np.mean(np.diff(r_hr)))
        HR_ans.append(hr)
        peak_locs_1.append(np.asarray(peaks))
    R_ans = peak_locs_1### Determines which of the peak locations is used for computing the score.)
    return R_ans,HR_ans

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath',type  = str , help = 'Path to the dataset')
    parser.add_argument('--refpath', type = str, help = 'Path to the file containing r peak location')
    parser.add_argument('--algorithm', type = int, help = 'Path to the file containing r peak location')
    parser.add_argument('--device', type = str , default = 'cpu', help = 'cuda / cpu')
    parser.add_argument('--model_path', type = str , help = 'Path to the DL Model')
    args = parser.parse_args()
    return args

def main():

    args= argparser() 
    fs_ = 500
    thr_ = 0.075
    only_test_data = True

    patient_ecg,patient_ref,R_ref,HR_ref  = load_patient_data(args.datapath,args.refpath,fs_)
   
    if args.algorithm == 3:
        windowed_data = patient_ecg[1936:,:]
        windowed_beats  = patient_ref[1936:] 
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(np.asarray(windowed_data).transpose()).transpose()
        BATCH_SIZE = 16 
        patient_ecg_t = torch.from_numpy(scaled_data).float() 
        patient_ecg_t = patient_ecg_t.view((patient_ecg_t.shape[0],1,patient_ecg_t.shape[1]))
        patient_ecg_tl = TensorDataset(patient_ecg_t)
        trainloader = DataLoader(patient_ecg_tl, batch_size=BATCH_SIZE)
        SAVED_MODEL_PATH = args.model_path                                                                                                                             
        y_pred = load_model_CNN(SAVED_MODEL_PATH,trainloader,args.device)
        y_pred_1 = [] 
        for batch in range(len(y_pred)):
            for record in range(len(y_pred[batch])):
                y_pred_1.append(y_pred[batch][record].cpu().numpy())
        y_pred_array = np.asarray(y_pred_1) 
        y_pred_array_1 = np.asarray(y_pred_1)
        peak_locs = []
        for i in range(y_pred_array.shape[0]):
            peak_locs.append(scipy.signal.find_peaks(-y_pred_array[i,:],distance = 45,height = -0.2,prominence = 0.035)[0])
        
        ### Getting the amplitude values at valley location. 
        y_roll_valleys = []
        y = []
        for j in range(len(peak_locs)):
            y = [y_pred_array[j,i] for i in peak_locs[j]]
            y_roll_valleys.append(y)

        ### Calling the scoring Function    
   
    else:
        peak_locs,results = call_detector(args,patient_ecg,patient_ref,R_ref,fs_)
        
    r_ans,hr_ans = compute_HR(peak_locs,fs_)
    if(only_test_data):
        r_ref,hr_ref = R_ref[1936:],HR_ref[1936:] 
        rec_acc,hr_acc = score(r_ref, hr_ref, r_ans, hr_ans, fs_, thr_)
    else:
        rec_acc,hr_acc = score(r_ref, hr_ref, r_ans, hr_ans, fs_, thr_)

if __name__ == "__main__":
    main()
