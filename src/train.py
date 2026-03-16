import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json

def train():
    df = pd.read_csv('data/housing.csv')
    
    df = df.dropna()
    
    df = pd.get_dummies(df, columns=['ocean_proximity'])

    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {
        "Dataset size": len(df),
        "RMSE": rmse,
        "R2": r2
    }
    
    print(f"Dataset Size: {len(df)}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    train()