export INPUT_FILE=./data/creditcard_2023.csv
export WEIGHT_FILE=./weights/pca.json
export CONFIG=./config/pca.yaml

python main.py \
    --input_file ${INPUT_FILE} \
    --algorithm PCA \
    --weights_save_file ${WEIGHT_FILE} \
    --config_file ${CONFIG}