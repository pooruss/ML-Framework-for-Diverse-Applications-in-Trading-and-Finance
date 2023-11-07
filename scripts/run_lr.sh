export INPUT_FILE=./data/boston_house_prices.csv
export WEIGHT_FILE=./weights/lr.json
export CONFIG=./config/lr.yaml

python main.py \
    --input_file ${INPUT_FILE} \
    --algorithm LinearRegression \
    --weights_save_file ${WEIGHT_FILE} \
    --config_file ${CONFIG}