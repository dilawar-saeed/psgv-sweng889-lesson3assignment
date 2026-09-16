1. Implement the code to load the Seoul Bike Sharing Demand dataset into a pandas DataFrame from this source: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand. Show the first 5 rows, the DataFrame shape, and the column names. Keep the code simple. Only implement this stage. 

2. Analyze this dataset for a machine learning regression problem. Identify the target variable, the useful input features, the data types, possible data leakage, and how we should split the data. Also suggest simple success criteria for the model. Do not train a model yet. Keep the explanation short. 

3. Implement a small exploratory analysis for this DataFrame. Show summary statistics, missing values, duplicate rows, the target distribution, and a few simple relationships with Rented Bike Count. Use only a few useful plots. Add short comments explaining what each check tells us. Only implement this stage. 

4. Implement the code to clean and transform this DataFrame for modeling. Handle the Date column, missing values if any, duplicate rows if any, and categorical columns. Separate features X from target y. Use simple functions and add a short comment explaining the goal of each function. Only implement this stage. 


5. Implement simple feature engineering for this bike demand problem. Create useful features from Date and Hour, such as year, month, day of week, weekend, and time-of-day information. Keep only features that have a clear reason to help prediction. Briefly explain why each new feature may help. Only implement this stage. 


6. Choose one simple scikit regression model for this dataset and explain in 2 or 3 sentences why it is a good choice for this demo. Then implement the train/test split and model training code. Keep the code short and readable. Do not evaluate the model yet. Only implement this stage. 


7. Evaluate the trained regression model. Use MAE, RMSE, and R-squared. Briefly explain why each metric is useful. Implement the code to calculate the metrics and create one simple plot comparing actual and predicted values. Then help me interpret the results in plain English. Only implement this stage. 

I then used the generated code but a different chat window:

8. Inserted above is the code for an ML model I am using to predict bike demand. I need a simple streamlit UI, some additional python code to handle the app, and a file that connects my app to a locally hosted LLM that will review the prediction and make suggestions to the operator of the bike station.
