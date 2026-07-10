# =========== libraries ===========
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

# =========== functions ===========
def model_metrics(y_test, y_pred):

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))