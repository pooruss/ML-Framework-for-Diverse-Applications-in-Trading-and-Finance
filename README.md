# Framework for ML in Finance

## 📚 Background
Machine learning needs in the **finance** field: Machine learning algorithms can be used in many aspects such as risk management, asset management, market analysis and trading strategies, and have become a key tool in the field of finance and trading.
### Goals and Significance
#### Goals
* Develop a custom Python machine learning framework
* Intergrate basic machine learning algorithms implemented through packages such as numpy
* Evaluate the algorithms in a variety of financial scenarios
#### Significance
* Develop a machine learning framework for the financial field to provide solutions more suitable for financial problems instead of just relying on general machine learning libraries
* A framework for machine learning beginners to quickly get started with basic machine learning algorithms

## 📈 Applications
| Problem | Solution | Algorithm |
| :----: | :----: | :----:|
| Risk Management | Classification, Regression | Principle Component Analysis(PCA), Adaboost|
| Financial Fraud Detection | Classification, Clustering | K-Nearest Neighbor(KNN), K-Means, DBSCAN |
| Customer Relationship Management | Classification | Naive Bayes, Adaboost |
| Financial Forecast | Regression | Support Vector Machine(SVM) |
| Investment and Asset Management | Regression | Linear Regression |

## 🎮 Other Machine learning Algorithms
- Multilayer Perceptron, used for classification / regression, can be applied to promotion, fraud detection and so on.
- Decision Tree, used for classification, can be applied to direct marketing, risk management and so on.
- ...

## 🎬 Demo
Here is a demo of using svm on the netflix stock dataset to do finance prediction.




### Main Content
✨Here is an overview of this framework.
<br>
<div align="center">
<img src="assets/overview.jpg" width="400px">
</div>
<br>

## Setup
- Install.
```bash
pip install -i requirements.txt
```

## Run
- Init weight.
Create a model config yaml file under `./config/`, which indicate the initial weight name and value of the model. Examples can be found in the existing config directory.
- Write the bash command under `./scripts/`. Examples can be found in the existing config directory.
```bash
bash scripts/run_svm.sh
```
During running, you need to enter natural language that describe how you would like to preprocess the data. After model training, you also need to enter the evaluation metric and the visualization method you would use.


## Main File Sturcture
```
├── main.py
├── pipeline.py
├── ...
├── /dataset/
│  ├── base.py
│  ├── risk_management.py
│  ├── investment_and_asset_management.py
│  ├── ...
│  └── finance_prediction.py
├── /algorithms/
│  ├── base.py
│  ├── svm.py
│  ├── linear_regression.py
│  ├── ...
│  └── pca.py
├── /evaluate/
│  └── utils.py
├── /visualization/
│  └── utils.py
```

## Results and Evaluation

Common financial problems that can be solved by applying machine learning methods can be categorized into the following five categories: financial fraud detection, customer relationship management, financial forecasting, risk management, investment and asset management. In each class of problems, a suitable dataset as well as a reasonable method was selected for testing and validation, and the results are as follows.

### Financial fraud detection

- Dataset: Credit Card Fraud Detection Dataset 2023
- Algorithms: k-Nearest Neighbor

The K-nearest neighbors (KNN) algorithm applied to classify credit card fraud dataset achieved an accuracy of 78.9%. With precision values of 74.2% for identifying non-fraudulent transactions and 86.0% for detecting fraudulent transactions, the algorithm demonstrates good performance in accurately predicting both classes. The recall values of 88.9% for non-fraudulent transactions and 68.9% for fraudulent transactions indicate that the algorithm effectively captures a high proportion of both classes. Overall, the KNN algorithm shows promising performance in accurately classifying credit card fraud, with balanced precision and recall values.

### Customer relationship management

- Dataset: Credit Card Customer Churn Prediction
- Algorithms: Naive Bayes

The Naive Bayes algorithm for predicting customer churn achieved an accuracy of 82.1%, with precision values of 85.1% for non-churned customers and 57.3% for churned customers. However, the recall values were 94.2% for non-churned customers and 32.1% for churned customers, indicating room for improvement in accurately identifying churned customers.

<img src="https://lh7-us.googleusercontent.com/Xu53AatzXsgwPCTT3kf5gd487I20fQ3s91L214T9yRYYqLxcSas2VKoLFoFYUZs63Lc4bI77A0hRGEePMUsn-kb9luRPwkNY5WxX9udLv3UjIi65fwuaAxwIi1SPeFYHeFvT8hhXJRfmhfcpvVvSGVtHcQ=s2048" alt="img" style="zoom:50%;" />

### Financial forecasting

- Dataset: Stock Price of Netflix
- Algorithms: SVM

Given history volumns, predict if a stock’s open price is higher or lower than the close price. The label is 0 if the close price is higher than the open price, and is 1 if the open price is higher than the close price. Predictions were 56% accurate. However, all predictions are label 1, might be overfitted.

<img src="https://lh7-us.googleusercontent.com/aIZ8Byl5ZxrOup5KeE67L8sVZ7UWxagSSnLr0JK9hcCAqoNRIi4fS6RWxROyUinMr3idm0Wppv9_o6WupyIr4r6OZCZ9PifcV3mrN7thwyBkMhtUx3vpfELMOHbRJ9D5xegcUpivQnYSuI0lUOx2qC_1wg=s2048" alt="img" style="zoom:50%;" />

### Risk management

### Investment and asset management

## Future Work

* Develop interactive UI
* Intergrate More algorithms
* Encapsulate the data preprocessing process to reduce the cost of getting started
