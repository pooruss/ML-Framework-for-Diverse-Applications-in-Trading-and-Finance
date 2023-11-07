export INPUT_FILE=./data/credit_card_demo.csv
export WEIGHT_FILE=./weights/adaboost.json
export CONFIG=./config/svm.yaml

python main.py \
    --input_file ${INPUT_FILE} \
    --algorithm AdaBoost \
    --weights_save_file ${WEIGHT_FILE} \
    --config_file ${CONFIG}