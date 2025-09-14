# Data_Science_Machine_Learning Project_Manufacturing_data

# Objective: 
The goal of this project is to enhance the testing approach in Mercedes-Benz’s manufacturing pipeline by lowering the time a car takes on the test bench. With given dataset containing various configurations of vehicle features, the project is to develop a forecasting model which can predict the time expected for each configuration to cross quality evaluations. The procedure includes many data preprocessing steps which include variance analysis for feature selection, resolve missing values, employ label encoding, and perform dimensionality reduction.  By adopting the XGBoost model, the project aims to give a reliable and effective solution which supports rapid testing, lowered costs, and improved long-term viability in line with Mercedes-Benz’s commitment to cutting-edge solutions, and sustainable practices.

# Dataset:
Train Data: train.csv
Test Data: test.csv
Data file: “Data_Science_Machine_Learning Project_Manufacturing_data.ipynb”
Predicted Results: predictions_data.csv: Model produced predictions of test dataset
Columns with zero variance or duplicates were removed. Missing values were checked.

# Methods:
Data cleaning: dropped zero variance columns, checked for same columns in train and test datasets.
Encoding categorical features: applied labelencoder to change categorical columns into numeric columns. 
Scaling and dimensionality reduction: used standardscaler to standardize features, decreased dimensions using PCA to keep 92% of variance for efficiency.
Modeling: train the XGBoost to predict evaluation time, if XGBoost is not available, use GradientBoostingRegressor.
Validation: 6-fold cross-validation was performed, RMSE was calculated to judge model accuracy

# Tools and technologies:
Python, jupyter notebook NumPy, pandas, scikit-learn, LabelEncoder, xgboost, GradientBoostingRegressor, Cross-validation (6-fold), Root Mean Squared Error (RMSE) evaluation, Dimensionality Reduction (PCA), working missing values, zero variance columns, feature scaling, encoding categorical variables, data preprocessing

# Results:
Cross-validation RMSE: mean RMSE: ~9.42 and std deviation: ~1.05
Sample Predictions: The model predicts test times, showing faster and optimized production management. 

# Conclusion: 
In this project, python machine learning techniques were used to a real manufacturing fine tuning problem.After cleaning the dataset, encoding the dataset, dropping irrelevant features, and implementing dimensionality minimization, XGBoost was used to train the model to get reliable forecasts of evaluation time. Acceptable results were obtained from the data preprocessing with an efficient model.  With model accuracy, the project decreases environmental impact and supports large scale industrial innovation. This end-to-end pipeline shows the real skills which are necessary to solve advanced problems in the automobile and manufacturing industries.

