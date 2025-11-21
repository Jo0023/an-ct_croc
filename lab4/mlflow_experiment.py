import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def recreate_experiment():
    print('Recreating iris-classification-tp4 experiment...')
    
    mlflow.set_tracking_uri('http://mlflow.labs.itmo.loc:5000')
    mlflow.set_experiment('iris-classification-tp4')
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Run 1: Configuration de base
    with mlflow.start_run(run_name='recreated-run-1'):
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        mlflow.log_params({'n_estimators': 100, 'max_depth': 5, 'dataset': 'iris'})
        mlflow.log_metric('accuracy', accuracy)
        print(f'Run 1 - Accuracy: {accuracy:.4f}')
    
    # Run 2: Moins d'arbres, profondeur réduite
    with mlflow.start_run(run_name='recreated-run-2'):
        model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        mlflow.log_params({'n_estimators': 50, 'max_depth': 3, 'dataset': 'iris'})
        mlflow.log_metric('accuracy', accuracy)
        print(f'Run 2 - Accuracy: {accuracy:.4f}')
    
    # Run 3: Plus d'arbres, profondeur augmentée
    with mlflow.start_run(run_name='recreated-run-3'):
        model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        mlflow.log_params({'n_estimators': 150, 'max_depth': 7, 'dataset': 'iris'})
        mlflow.log_metric('accuracy', accuracy)
        print(f'Run 3 - Accuracy: {accuracy:.4f}')
    
    print('Experiment recreated with 3 runs!')
    print('Refresh MLflow interface to see the results.')

if __name__ == "__main__":
    recreate_experiment()