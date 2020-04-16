BASE_PATH='../../../Datasets/ECG'
DATAPATH=${BASE_PATH}'/nstdb/' 
DEVICE='cuda'
MODEL_PATH='../model_1.pt'

echo python evaluate_nstdb.py --datapath ${DATAPATH} --evaluate_nstdb --device ${DEVICE} --model_path ${MODEL_PATH}
python evaluate_nstdb.py --datapath ${DATAPATH} --evaluate_nstdb --device ${DEVICE} --model_path ${MODEL_PATH}
