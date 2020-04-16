#DATASET='mitdb'
DATASET='nstdb'
#DATASET='mit_bih_Exercise_ST_change'
BASE_PATH='../../../Datasets/ECG'
DATAPATH=${BASE_PATH}'/'${DATASET}'/' 
DEVICE='cuda'
MODEL_PATH='../model_1.pt'

echo python evaluate.py --dataset ${DATASET} --datapath ${DATAPATH} --device ${DEVICE} --model_path ${MODEL_PATH}
python evaluate_datasets.py --dataset ${DATASET} --datapath ${DATAPATH} --device ${DEVICE} --model_path ${MODEL_PATH}

