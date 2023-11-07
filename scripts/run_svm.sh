export INPUT_FILE=./data/NFLX.csv
export WEIGHT_FILE=./weights/svm.json
export CONFIG=./config/svm.yaml

python main.py \
    --input_file ${INPUT_FILE} \
    --algorithm SimpleSVM \
    --weights_save_file ${WEIGHT_FILE} \
    --config_file ${CONFIG}