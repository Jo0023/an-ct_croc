import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import json
import os

print("=== IRIS TRAINING WITH ARTIFACTS ===")

# Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris-Classification-With-Artifacts")

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names  # This is already a list
target_names = iris.target_names    # This is already a list

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

with mlflow.start_run(run_name="RandomForest-With-Artifacts"):
    # Train model
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predictions and metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    
    # Log parameters
    mlflow.log_params(params)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("train_samples", X_train.shape[0])
    mlflow.log_metric("test_samples", X_test.shape[0])
    
    # Log model as artifact - THIS IS WHAT WAS MISSING
    mlflow.sklearn.log_model(model, "model")
    
    # Log additional artifacts
    # 1. Feature importance
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    with open("/tmp/feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)
    mlflow.log_artifact("/tmp/feature_importance.json", "model_info")
    
    # 2. Dataset info (CORRECTED - no .tolist() needed)
    dataset_info = {
        "feature_names": feature_names,  # Already a list
        "target_names": target_names.tolist(),  # This one needs .tolist()
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape)
    }
    with open("/tmp/dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    mlflow.log_artifact("/tmp/dataset_info.json", "dataset")
    
    # 3. Training configuration
    config = {
        "model_type": "RandomForestClassifier",
        "framework": "scikit-learn",
        "test_size": 0.2,
        "random_state": 42
    }
    with open("/tmp/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    mlflow.log_artifact("/tmp/training_config.json", "config")
    
    print("Model and artifacts logged successfully!")

print("=== TRAINING COMPLETED ===")