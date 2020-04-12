BASE_PATH='../../Datasets/ECG'
DATAPATH=${BASE_PATH}'/nstdb/' 

echo python evaluate_nstdb.py --datapath ${DATAPATH} --evaluate_nstdb
python evaluate_nstdb.py --datapath ${DATAPATH} --evaluate_nstdb
