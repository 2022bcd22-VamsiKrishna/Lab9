import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import json

def train():
    # 1. Connect to your local MLflow server on port 8081
    mlflow.set_tracking_uri("http://localhost:8081")
    
    # 2. STRICT IDENTITY REQUIREMENT: Your Name and Roll Number
    mlflow.set_experiment("Vamsi_2022BCD0022_Experiment")

    # 3. Start the MLflow Run
    with mlflow.start_run(run_name="RandomForest_Run1"):
        df = pd.read_csv('data/housing.csv')
        df = df.dropna()
        df = pd.get_dummies(df, columns=['ocean_proximity'])

        X = df.drop('median_house_value', axis=1)
        y = df['median_house_value']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Log Parameter
        n_estimators = 100
        mlflow.log_param("n_estimators", n_estimators)

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Log Metrics
        mlflow.log_metric("Dataset_size", len(df))
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log Model Artifact
        mlflow.sklearn.log_model(model, "random_forest_model")

        print(f"Dataset Size: {len(df)}")
        print(f"RMSE: {rmse}")
        print(f"R2 Score: {r2}")
        print("Logged to MLflow successfully!")

if __name__ == '__main__':
    train()