import numpy as np
from colorama import init, Fore, Style
from dataset import BaseDataset
from algorithms import BaseAlgorithm
from evaluate.utils import (
    get_accuracy,
    MSE,
    r2_score,
    MAE,
    RMSE,
    silhouette_coefficient,
    supported_metrics
)
from visualization.utils import (
    confusion_matrix,
    plotscatter2d,
    plotscatter3d,
    decision_boundary,
    supported_visuals
)

init(autoreset=True)

class Pipeline:
    """
    This class will use the Dataset and SimpleSVM classes to train the SVM model and make predictions.
    """
    def __init__(self, dataset: BaseDataset, model: BaseAlgorithm):
        self.dataset = dataset
        self.model = model

    def run(self, weights_save_file):
        print(f"{Fore.GREEN}Running pipeline with model {self.model} and dataset {self.dataset.file_path}.{Style.RESET_ALL}")
        # Train the model
        X_train, y_train = self.dataset.get_train_data()
        print(f"{Fore.YELLOW}Training...{Style.RESET_ALL}")
        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            # TODO add reflection or support user to modify the wrong code.
            pass
        print(f"{Fore.GREEN}Finish training.{Style.RESET_ALL}")
        self.model.save_params(weights_save_file)

        # Make predictions
        X_test, y_test = self.dataset.get_test_data()
        print(f"{Fore.YELLOW}Predicting...{Style.RESET_ALL}")
        predictions = self.model.predict(X_test)
        print(f"{Fore.GREEN}Finish predicting.{Style.RESET_ALL}")
        return predictions
    
    def evaluation(self, predictions=None, labels=None, *kwargs):
        method = input(f"{Fore.RED}Enter your evaluation metrics.\nThe following metrics are supported:{supported_metrics}(enter exit to skip)\n{Style.RESET_ALL}")
        if method == "accuracy":
            result = get_accuracy(predictions, labels)
        elif method == "MSE":
            result = MSE(predictions, labels)
        elif method == "r2 score":
            result = r2_score(predictions, labels)
        elif method == "MSE":
            result = MSE(predictions, labels)
        elif method == "MAE":
            result = MAE(predictions, labels)
        elif method == "RMSE":
            result = RMSE(predictions, labels)
        elif method == "silhouette coefficient":
            result = silhouette_coefficient(predictions, labels)
        else:
            raise NotImplementedError
        return result
    
    def visualization(self, predictions=None, labels=None, *kwargs):
        method = input(f"{Fore.RED}Enter your visualization method.\nThe following metrics are supported:{supported_visuals}(enter exit to skip)\n{Style.RESET_ALL}")
        if method == "confusion matrix":
            result = confusion_matrix(predictions, labels)
        elif method == "plot scatter 2d":
            result = plotscatter2d(predictions, labels)
        elif method == "plot scatter 3d":
            result = plotscatter3d(predictions, labels)
        elif method == "decision boundary":
            result = decision_boundary(predictions, labels, *kwargs)

        else:
            raise NotImplementedError
        return result
        

        

    