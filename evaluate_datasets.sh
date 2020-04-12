#DATASET='mitdb'
DATASET='nstdb'
#DATASET='mit_bih_Exercise_ST_change'
BASE_PATH='../../Datasets/ECG'
DATAPATH=${BASE_PATH}'/'${DATASET}'/' 

echo python evaluate.py --dataset ${DATASET} --datapath ${DATAPATH}
python evaluate_datasets.py --dataset ${DATASET} --datapath ${DATAPATH}
