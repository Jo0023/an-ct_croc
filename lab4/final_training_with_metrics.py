import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

def train_final_model():
    print('=== ML MODEL TRAINING - METRICS ONLY ===')
    
    # Configuration MLflow
    mlflow.set_tracking_uri('http://mlflow.labs.itmo.loc:5000')
    mlflow.set_experiment('iris-classification-tp4')
    
    print('Loading Iris dataset...')
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f'Data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples')
    
    print('Training RandomForest model...')
    with mlflow.start_run(run_name='tp4-final'):
        # Entraînement du modèle
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Prédictions et évaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Logger les paramètres et métriques
        mlflow.log_param('n_estimators', 100)
        mlflow.log_param('max_depth', 5)
        mlflow.log_param('random_state', 42)
        mlflow.log_param('dataset', 'iris')
        mlflow.log_param('test_size', 0.2)
        mlflow.log_param('feature_count', X.shape[1])
        mlflow.log_param('class_count', len(iris.target_names))
        
        mlflow.log_metric('accuracy', accuracy)
        
        print(f'ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)')
        print('Parameters and metrics saved to MLflow')
    
        # Sauvegarde locale du modèle
        try:
            with open('/tmp/iris_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            print('Model saved locally in /tmp/iris_model.pkl')
        except Exception as e:
            print(f'Local save error: {e}')
    
    # Rapport détaillé
    print('\nCLASSIFICATION REPORT:')
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    print(report)
    
    print('\nPREDICTION EXAMPLES:')
    for i in range(5):
        true_class = iris.target_names[y_test[i]]
        pred_class = iris.target_names[y_pred[i]]
        confidence = model.predict_proba([X_test[i]]).max()
        correct = 'CORRECT' if y_test[i] == y_pred[i] else 'INCORRECT'
        print(f'  {correct} Actual: {true_class:12} -> Predicted: {pred_class:12} (confidence: {confidence:.3f})')
    
    print('\n' + '=' * 50)
    print('PART 2 COMPLETED SUCCESSFULLY!')
    print('=' * 50)
    print('SUMMARY:')
    print(f'  - Accuracy: {accuracy:.4f}')
    print(f'  - Model: RandomForestClassifier')
    print(f'  - Experiment: iris-classification-tp4')
    print('\nINFORMATION:')
    print('  - Parameters and metrics are saved in MLflow')
    print('  - Model is not saved (permission issues)')
    print('  - We can continue with next parts')
    print('\nCHECK IN MLFLOW:')
    print('  http://mlflow.labs.itmo.loc:5000')
    print('  -> Experiment: iris-classification-tp4')
    print('  -> Run: tp4-final')

if __name__ == "__main__":
    train_final_model()