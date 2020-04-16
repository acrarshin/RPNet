#ALGORITHM=0
#ALGORITHM=1
#ALGORITHM=2
ALGORITHM=3
DATAPATH="../../CPSC_2019/train/data"
REFPATH="../../CPSC_2019/train/ref"
DEVICE='cuda'
MODEL_PATH='../model_1.pt'

echo python evaluate_detectors_CPSC.py --dataset ${DATAPATH} --refpath ${REFPATH} --algorithm ${ALGORITHM} --device ${DEVICE} --model_path ${MODEL_PATH}
python evaluate_detectors_CPSC.py --datapath ${DATAPATH} --refpath ${REFPATH} --algorithm ${ALGORITHM} --device ${DEVICE} --model_path ${MODEL_PATH}
