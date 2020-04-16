import numpy as np
from tqdm import tqdm
from glob import glob
import wfdb as wf
import scipy.signal
from sklearn.preprocessing import StandardScaler
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from network import IncUNet

def obtain_data(args):
    #path = '../../Datasets/ECG/mitdb/'
    dat_path = args.datapath + '*.atr'
    paths = glob(dat_path)
    p2 = [path[:-4] for path in paths]
    fs = wf.rdsamp(p2[0])[1]['fs']
    windowed_data = []
    windowed_beats = []
    count = 0
    count1 = 0
    
    for path in tqdm(p2):
        ann    = wf.rdann(path,'atr')
        record = wf.io.rdrecord(path)
        beats  = ann.sample
        labels = ann.symbol
        len_beats = len(beats)
        data = record.p_signal[:,0]
        if(args.dataset == 'mitdb' or args.dataset == 'nstdb'):
            ### Sampling rate in  MITDB and NSTDB is 360. Therefore, for 10s data the window size is 3600 ###
                
                if (path[-2:] == '_6' or (path[-2:] != str(args.db) and args.evaluate_nstdb)):#(path[-2:] == '_6' or path[-2:] == '12' or path[-2:] == '06' or path[-2:] == '18' or path[-2:] == '24'):
                    print('Skip')
                    continue
                else:
                    ini_index = 0
                    final_index = 0
                    ### Checking for Beat annotations
                    non_required_labels = ['[','!',']','x','(',')','p','t','u','`',"'",'^','|','~','+','s','T','*','D','=','"','@']
                    for window in range(len(data) // 3600):
                        count += 1
                        for r_peak in range(ini_index,len_beats):
                            if beats[r_peak] > (window+1) * 3600:
                                final_index = r_peak
                                #print('FInal index:',final_index)
                                break
                        record_anns = list(beats[ini_index: final_index])
                        record_labs = labels[ini_index: final_index]
                        to_del_index = []
                        for actual_lab in range(len(record_labs)):
                            for lab in range(len(non_required_labels)):
                                if(record_labs[actual_lab] == non_required_labels[lab]):
                                    to_del_index.append(actual_lab)
                        print('To del Indices are:',to_del_index)
                        for indice in range(len(to_del_index)-1,-1,-1):
                            print(indice)
                            del record_anns[to_del_index[indice]]
                        windowed_beats.append(np.asarray(record_anns) - (window) * 3600)
                        windowed_data.append(data[window * 3600 : (window+1) * 3600])
                        ini_index = final_index
        elif(args.dataset == 'mit_bih_Exercise_ST_change'):
            ini_index = 0
            final_index = 0
            for window in range(len(data) // 3600):
                count += 1
                windowed_data.append(data[window * 3600 : (window+1) * 3600])
                for r_peak in range(ini_index,len_beats):
                    if beats[r_peak] > (window+1) * 3600:
                        final_index = r_peak
                        #print('FInal index:',final_index)
                        break
                windowed_beats.append(beats[ini_index: final_index] - (window) * 3600)
                ini_index = final_index    
    
    ### Scaling and Resampling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.asarray(windowed_data).transpose()).transpose()
    resampled_beat = []
    for record in range(len(windowed_data)):
        resampled_beat.append(scipy.signal.resample(scaled_data[record],5000))
    patient_ecg = np.asarray(resampled_beat)
    
    return patient_ecg,windowed_beats


def score(r_ref, r_ans, fs_, thr_):
    for record in range(len(r_ref)):
        r_ref[record] = r_ref[record][(r_ref[record] >= 0.5*fs_) & (r_ref[record] <= 9.5*fs_)]
    all_TP = 0
    all_FN = 0
    all_FP = 0
    failed_record = []
    fp = []
    fn = []
    zer = []
    HR_score = 0
    record_flags = np.ones(len(r_ref))
    failed_record = []
    fp = []
    fn = []
    zer = []
    for i in range(len(r_ref)):
        FN = 0
        FP = 0
        TP = 0
#         if math.isnan(hr_ans[i]):
#             hr_ans[i] = 0
#         hr_der = abs(int(hr_ans[i]) - int(hr_ref[i]))
#         if hr_der <= 0.02 * hr_ref[i]:
#             HR_score = HR_score + 1
#         elif hr_der <= 0.05 * hr_ref[i]:
#             HR_score = HR_score + 0.75
#         elif hr_der <= 0.1 * hr_ref[i]:
#             HR_score = HR_score + 0.5
#         elif hr_der <= 0.2 * hr_ref[i]:
#             HR_score = HR_score + 0.25
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
        
        ##Custom
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

    
    print("Failed_records:",failed_record)
    print("FP's:",fp)
    print("FN's:",fn)
    print("Zeros's:",zer)
    ##
    rec_acc = round(np.sum(record_flags) / len(r_ref), 4)
    hr_acc = round(HR_score / len(r_ref), 4)

    print( 'QRS_acc: {}'.format(rec_acc))
    print('HR_acc: {}'.format(hr_acc))
    print('Scoring complete.')
    Recall = all_TP / (all_FN + all_TP)
    Precision = all_TP / (all_FP + all_TP)
    F1_score = 2 * Recall * Precision / (Recall + Precision)
    print("TP's:{} FN's:{} FP's:{}".format(all_TP,all_FN,all_FP))
    print("REcall:{}, Precision(FNR):{}, F1-Score:{}".format(Recall,Precision,F1_score))
    return rec_acc,all_FP,all_FN,all_TP

def load_model_CNN(SAVED_MODEL_PATH,test_loader,device='cpu'):
    C,H,W = 1,1,5000
    loaded_model = IncUNet(in_shape=(C,H,W))
    loaded_model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location = lambda storage, loc: storage, pickle_module=pickle))
    loaded_model.to(device)
    loaded_model.eval()
    print("...........Evaluation..........")
    loaded_model.eval()
    ### Need to change after this ###
    net_test_loss = 0   
    y_pred = []
    batch_length = 64
    y_pred = []
    with torch.no_grad():
        for step,x in enumerate(test_loader):
            print('Step = ',step)
            x = Variable(x[0].to(device))
            y_predict_test = loaded_model(x)
            y_pred.append(y_predict_test[:,0,:])                    
    return y_pred

