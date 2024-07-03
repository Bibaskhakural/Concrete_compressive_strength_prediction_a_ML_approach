# This script can contain the main workflow of your project

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def train_model(data):
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(),
    'Ridge Regression': Ridge(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'Neural Network Regressor': MLPRegressor(random_state=42)
    }
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
   results = {}
   for model_name, model in models.items():
      mae, rmse, r2 = model_evaluation(model, X_train, y_train, X_test, y_test)
      results[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2 Score': r2}

    results_df = pd.DataFrame(results).T
    print("Model Performance:")
    print(results_df)
    
    return model

if __name__ == '__main__':
    data = load_data('data/processed/dataset.csv')
    model = train_model(data)
