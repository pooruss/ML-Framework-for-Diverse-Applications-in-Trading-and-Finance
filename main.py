import argparse
from dataset.base import BaseDataset
from pipeline import Pipeline
from algorithms import (
    SimpleSVM, 
    AdaBoost,
    ID3,
    MLP,
    DBSCAN,
    PCA,
    LinearRegression,
    NBNumber,
    KMeans
)

def main(args):
    input_file = args.input_file
    algorithm = args.algorithm
    weights_save_file = args.weights_save_file
    # Instantiate the Dataset
    dataset = BaseDataset(input_file)
    
    # Choose the algorithm
    if algorithm == 'SimpleSVM':
        model = SimpleSVM(args.config_file)
    elif algorithm == 'AdaBoost':
        model = AdaBoost(args.config_file)
    elif algorithm == 'ID3':
        model = ID3(args.config_file)
    elif algorithm == 'MLP':
        model = MLP(args.config_file)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(args.config_file)
    elif algorithm == 'PCA':
        model = PCA(args.config_file)
    elif algorithm == 'LinearRegression':
        model = LinearRegression(args.config_file)
    elif algorithm == 'NBNumber':
        model = NBNumber(args.config_file)
    elif algorithm == 'KMeans':
        model = KMeans(args.config_file)
    
    else:
        raise ValueError(f"Algorithm {algorithm} is not supported. Choose 'SimpleSVM' or 'AdaBoost'.")
    
    # Instantiate the Pipeline and run it
    pipeline = Pipeline(dataset, model)
    predictions = pipeline.run(weights_save_file=weights_save_file)
    
    # Evaluation
    _, y_test = dataset.get_test_data()
    evaluation_result = pipeline.evaluation(predictions, y_test)
    print(f"The evaluation result of {algorithm} on the test set is: {evaluation_result:.2f}")
    
    # Visualization
    visualization_result = pipeline.visualization(predictions, y_test)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the SVM or AdaBoost algorithm on a dataset.')
    parser.add_argument('--config_file', type=str, required=True, help='The path to config file.')
    parser.add_argument('--input_file', type=str, required=True, help='The path to the CSV file containing the data.')
    parser.add_argument('--weights_save_file', type=str, required=True, help='The path to save the trained model.')
    parser.add_argument('--algorithm', type=str, required=True, help='The name of the algorithm to use.')
    
    args = parser.parse_args()
    main(args)