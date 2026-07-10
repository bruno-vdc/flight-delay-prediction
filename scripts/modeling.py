# %%
# =========== libraries ===========
import pandas as pd
from pathlib import Path
import config as CONFIG
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import models.logistic_regression as mlogreg
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import dagshub
import mlflow
import mlflow.sklearn

#setting pandas options
pd.set_option("display.max_columns", None) #columns
pd.set_option("display.max_rows", 10)      #rows

# =========== setting dagshub and mlflow ===========
dagshub.init(repo_owner="bruno-vdc", repo_name="flight-delay-prediction", mlflow=True)
mlflow.set_experiment("Flight Delay Prediction")

# =========== paths and files ===========
PROJECT_ROOT = Path(__file__).resolve().parent.parent

flights_featured = pd.read_csv(PROJECT_ROOT/CONFIG.FEATURED_DATASET_PATH)

#=========== model fit ===========
X = flights_featured[CONFIG.PREDICTOR_LIST]
y = flights_featured[CONFIG.TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=CONFIG.TEST_SIZE, random_state=CONFIG.RANDOM_STATE)

#=========== encoding ===========
#ordinal encoder
ord_enc = OrdinalEncoder(categories=[['Very Short Haul', 'Short Haul', 'Medium Haul', 'Long Haul']])

#onehot encoder
oh_enc = OneHotEncoder(handle_unknown='ignore')

#standard scaler
ss_enc = StandardScaler()

#column transformer
col_tran = ColumnTransformer([("nominal", oh_enc, CONFIG.NOMINAL_FEATURES),
                              ("ordinal", ord_enc, CONFIG.ORDINAL_FEATURES),
                              ("numerical", ss_enc, CONFIG.NUMERICAL_FEATURES)])

#fitting and transforming
X_train_tran = col_tran.fit_transform(X_train)

X_test_tran = col_tran.transform(X_test)

# %%
#=========== models run ===========
#logistic regression
with mlflow.start_run(run_name="Baseline Logistic Regression - Scaled"):
    
    model_log_reg = mlogreg.train(X_train_tran, y_train)

    y_log_reg = model_log_reg.predict(X_test_tran)

    mlflow.log_param("model", mlogreg.MODEL_NAME)
    mlflow.log_param("max_iter", mlogreg.MAX_ITER)
    mlflow.log_param("solver", mlogreg.SOLVER)
    mlflow.log_param("class_weight", mlogreg.CLASS_WEIGHT)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_log_reg))
    mlflow.log_metric("precision", precision_score(y_test, y_log_reg))
    mlflow.log_metric("recall", recall_score(y_test, y_log_reg))
    mlflow.log_metric("f1_score", f1_score(y_test, y_log_reg))
    
    mlflow.sklearn.log_model(model_log_reg, "model")

print(classification_report(y_test, y_log_reg))
print(confusion_matrix(y_test, y_log_reg))