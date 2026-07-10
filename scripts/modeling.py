# %%
# =========== libraries ===========
import pandas as pd
from pathlib import Path
import dagshub
import mlflow, mlflow.sklearn
import config as CONFIG
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import models.random_forest as MODEL
import evaluation.model_evaluation as EVAL
from sklearn.metrics import classification_report, confusion_matrix

# %%
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
                              ("numerical", ss_enc, CONFIG.NUMERICAL_FEATURES),
                              ("ordinal", ord_enc, CONFIG.ORDINAL_FEATURES)])

#fitting and transforming
X_train_tran = col_tran.fit_transform(X_train)

X_test_tran = col_tran.transform(X_test)

# %%
#=========== model run ===========
with mlflow.start_run(run_name="Random Forest - Deeper"):
    
    model = MODEL.train(X_train_tran, y_train, CONFIG.RANDOM_STATE)

    y_pred = model.predict(X_test_tran)

    mlflow.log_param("model", MODEL.MODEL_NAME)
    mlflow.log_param("n_estimatores", MODEL.N_ESTIMATORS)
    mlflow.log_param("max_depth", MODEL.MAX_DEPTH)
    mlflow.log_param("class_weight", MODEL.CLASS_WEIGHT)
    
    mlflow.log_param("min_samples_leaf", MODEL.MIN_LEAF)
    mlflow.log_param("min_samples_split ", MODEL.MIN_SPLIT)

    EVAL.model_metrics(y_test, y_pred)
    
    mlflow.sklearn.log_model(model, "model")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))