import wfdb as wf
import numpy as np
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import argparse

import torch
import torch.nn as n
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from py_ecg.ecgdetectors import Detectors

import py_ecg._tester_utils
from utils import score,load_model_CNN,obtain_data

def main(args): 
    
    patient_ecg,windowed_beats = obtain_data(args)
    
    BATCH_SIZE = 64
    patient_ecg_t = torch.from_numpy(patient_ecg).float()
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
    resampled_dt = []
    for record in range(y_pred_array.shape[0]):
        resampled_dt.append(scipy.signal.resample(y_pred_array_1[record],3600))
    y_pred_array = np.asarray(resampled_dt) 
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
    FS = 360
    THR = 0.075
    rec_acc,all_FP,all_FN,all_TP = score(windowed_beats,peak_locs, FS, THR)

def argparse_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default = 'nstdb',type  = str , help = 'Dataset Name')
    parser.add_argument('--datapath',type  = str , help = 'Path to the dataset')
    parser.add_argument('--db',default = 12,type = int,help = 'The DB of noise')
    parser.add_argument('--evaluate_nstdb',action = 'store_true',help = 'Mention this if you want to store action')
    parser.add_argument('--device', type = str , default = 'cpu' , help = 'cuda / cpu')
    parser.add_argument('--model_path', type = str , help = 'Path to the model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argparse_func() 
    main(args)
    pass
