
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=== DÉBUT ENTRAÎNEMENT IRIS ===")

# Configuration MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris-Classification")

# Charger les données
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Données chargées: {X.shape[0]} échantillons, {X.shape[1]} features")

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

with mlflow.start_run(run_name="RandomForest-Baseline"):
    # Entraînement du modèle
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Prédictions et métriques
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    # Logging des paramètres
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "test_size": 0.3
    })
    
    # Logging des métriques
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("train_samples", X_train.shape[0])
    mlflow.log_metric("test_samples", X_test.shape[0])
    
    # Logging du modèle
    mlflow.sklearn.log_model(model, "model")
    
    # Tags
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("dataset", "Iris")
    mlflow.set_tag("framework", "scikit-learn")
    
    print("Modèle entraîné et enregistré avec succès!")
    print(f"Accuracy finale: {accuracy:.4f}")