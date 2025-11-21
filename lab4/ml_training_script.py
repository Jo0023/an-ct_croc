import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    mlflow.set_tracking_uri('http://mlflow.labs.itmo.loc:5000')
    mlflow.set_experiment('iris-classification-tp4')
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name='tp4-final'):
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_params({
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42
            'dataset': 'iris'
        })
        mlflow.log_metric('accuracy', accuracy)
        mlflow.sklearn.log_model(model, 'model')
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
    
if __name__ == "__main__":
    train_model()