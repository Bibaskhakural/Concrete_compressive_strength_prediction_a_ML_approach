# This script can contain the main workflow of your project

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def train_model(data):
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    print(f'Mean Squared Error: {mse}')
    
    return model

if __name__ == '__main__':
    data = load_data('data/processed/dataset.csv')
    model = train_model(data)
