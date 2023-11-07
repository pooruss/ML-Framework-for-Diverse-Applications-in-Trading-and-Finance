import os
import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from openai_chat import OpenAIChat
from prompt import system_prompt, general_query
from utils import parse_solution
from sklearn.decomposition import PCA
from colorama import init, Fore, Style
init(autoreset=True)


class BaseDataset(ABC):
    """
    This class handles loading, preprocessing, and splitting of NFLX stock data.
    """
    def __init__(self, file_path, test_split=0.2, generated_codes_path="./cache/generated_codes.json"):
        self.file_path = file_path
        self.test_split = test_split
        self.generated_codes_path = generated_codes_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data = self.load_data()
        self.preprocessing(self.data)

    def load_data(self):
        # Load the data
        data = pd.read_csv(self.file_path)
        print(f"{Fore.GREEN}This is what your data look like: {Style.RESET_ALL}\n{Fore.YELLOW}{data}{Style.RESET_ALL}")
        return data
    
    def preprocessing(self, data, query=None):
        def exec_code(code_str: str): 
            exec(code_str) 
        if not query:
            query = input(f"{Fore.RED}\nState how you would like to preprocess your data, such as choose which attribute to be the Y label, whether to delete some unrelated X attributes, etc. If the data does not need preprocessing, enter skip. If use default policy, enter default:\n\n{Style.RESET_ALL}")
        if query == "skip":
            pass
        else:
            # use tmp_data.csv as input to the generated code
            data.to_csv("tmp_data.csv", index=False)

            # read from cached codes
            if os.path.exists(self.generated_codes_path):
                generated_codes = json.load(open(self.generated_codes_path, "r"))
            else:
                generated_codes = {}
            if query + f"\n{self.file_path}" in generated_codes:
                solution = generated_codes[query + f"\n{self.file_path}"]["code"]
                print("Using generated code.")
                exec_code(solution)
                preprocessed = True
            else:
                openai_chat = OpenAIChat()
                # set system prompt
                openai_chat.set_system_prompt(system_prompt)
                # modify the user prompt here.
                if query == "default":
                    query = general_query
                solution = openai_chat.chat(f"Here is the query:\n{query}\n\nHere is the data:\n{data}\nNow please write me the code, and your code should start with <CODE> and end with </CODE>.")
                solution = parse_solution(solution)
                print(f"{Fore.CYAN}\nThis is the code generated according to your request:\n{Fore.RESET}{solution}\n")
                try:
                    # execute the generated code to preprocess the data
                    exec_code(solution)
                    preprocessed = True

                except Exception as e:
                    print(f"{Fore.RED}\nError when executing the generated python code: {e}")
                    preprocessed = False

            if preprocessed:
                data = pd.read_csv('tmp_data.csv')
                os.system(f"mv tmp_data.csv {self.file_path}_preprocessed.csv")
                print(f"{Fore.YELLOW}\nThis is the preprocessed data:\n{Style.RESET_ALL}{data}\n")
                # cache the generated code
                self._store_code(query=query, solution=solution)


        X = data.iloc[:, :-1].values
        y = data.iloc[:,-1].values

        # Split into train/test set
        self._train_test_split(X, y)

    def _train_test_split(self, X, y):
        train_size = int((1 - self.test_split) * len(X))
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]
        # Add bias term
        self.X_train = np.hstack((np.ones((self.X_train.shape[0], 1)), self.X_train))
        self.X_test = np.hstack((np.ones((self.X_test.shape[0], 1)), self.X_test))

    def _store_code(self, query, solution):
        if os.path.exists(self.generated_codes_path):
            generated_codes = json.load(open(self.generated_codes_path, "r"))
        else:
            generated_codes = {}
        if query + f"\n{self.file_path}" not in generated_codes:
            generated_codes[query + f"\n{self.file_path}"] = {
                "query": query,
                "file": self.file_path,
                "code": solution
            }
            json.dump(generated_codes, open(self.generated_codes_path, "w"), indent=2)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def apply_pca_if_needed(self, data, n_components=10):
        if data.shape[1] > n_components:  # Check if the number of features is greater than 10
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(data)
            return reduced_data
        else:
            return data
        
    def feature_engineering(self, data):
        pass