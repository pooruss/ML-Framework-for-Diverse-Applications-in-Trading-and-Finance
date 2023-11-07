system_prompt = """You are an experienced python programmer and machine learning engineer which can write codes to fulfill user's requests for data preprocessing. 
You will be provided with a pandas DataFrame and a data preprocess request. Your task is to write a Python code to fulfill that request.
Note that:

- The code should take the 'tmp_data.csv' as input and save the preprocessed data as 'tmp_data.csv'. 
- You should move the Y variable (predict target) to the last column.
- Do not use sklearn.
"""

general_query = """1. Handle Missing Values: We need to decide how to handle missing values 2. Encode Categorical Variables: Machine learning algorithms require numerical input, so we'll need to encode categorical variables. Besides, data like Date should be converted into a more useful numerical format. Also, the predict target should be transformed into a numerical format since it's our target variable. 3.Feature Selection: Columns like ID is identifiers that are unlikely to have predictive power and can be removed. 5. Numerical Features: Ensure that all numerical features are in the correct format and scale if necessary."""

credit_card_query = """1. Handle Missing Values: We need to decide how to handle missing values, especially in the Gender column. 2. Encode Categorical Variables: Machine learning algorithms require numerical input, so we'll need to encode categorical variables like Gender and Merchant Name. We'll also need to transform the Category column into a numerical format since it's our target variable. 3.Feature Selection: Columns like Customer ID, Name, and Surname are identifiers that are unlikely to have predictive power and can be removed. 4. Convert Dates: The Birthdate and Date columns should be converted into a more useful numerical format. 5. Numerical Features: Ensure that all numerical features are in the correct format and scale if necessary."""

netflix_stock_query = """add a new column, the value if 0 if the close price is larger than the open price, and is 1 if the open prices is larger than the close price. Also, drop the Date and Adj Close column"""

