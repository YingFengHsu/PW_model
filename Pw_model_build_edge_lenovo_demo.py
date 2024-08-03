#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 05:36:58 2024

@author: ying-fenghsu
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Load your dataset (this is just sample file)
ML_data = pd.read_csv("sample_data.csv")

# Define the features (X) and the target (y)
X = ML_data.drop(columns='power')
y = ML_data['power']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=456)

# Define the model
xgboost_model = XGBRegressor(random_state=456)

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=xgboost_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best model
best_xgboost_model = grid_search.best_estimator_

# Print the best parameters
print("Best hyperparameters: ", grid_search.best_params_)

# Evaluate the model on the test set
y_pred = best_xgboost_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE) on test set: ", rmse)

# Results of the grid search
results = pd.DataFrame(grid_search.cv_results_)
print(results)

# Optionally, save the results to a CSV file
# results.to_csv('grid_search_results.csv', index=False)

#%% save model

# Save the model to a file
joblib.dump(best_xgboost_model, 'best_lenovo_xgboost_model.pkl')

print("Model saved to 'best_xgboost_model.pkl'")
