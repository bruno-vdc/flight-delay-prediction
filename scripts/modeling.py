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
import extra_features as EXTRA_FT
import models.logistic_regression as MODEL_LR
import models.random_forest as MODEL_RF
import models.xgboost as MODEL_XGB
import models.lightgbm as MODEL_LGBM
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

#original columns
original_cols = X_train.columns.tolist()

#=========== adding new features ===========
#route delay rate
X_train, X_test = EXTRA_FT.add_route_delay_rate(X_train, X_test, y_train, CONFIG.TARGET, EXTRA_FT.MIN_FLIGHTS)

#origin delay rate
X_train, X_test = EXTRA_FT.add_delay_rate_feature(X_train, X_test, y_train, CONFIG.TARGET, group_column="ORIGIN", feature_name="origin_delay_rate")

#destination delay rate
X_train, X_test = EXTRA_FT.add_delay_rate_feature(X_train, X_test, y_train, CONFIG.TARGET, group_column="DEST", feature_name="dest_delay_rate")

#airline delay rate
X_train, X_test = EXTRA_FT.add_delay_rate_feature(X_train, X_test, y_train, CONFIG.TARGET, group_column="AIRLINE_CODE", feature_name="airline_delay_rate")

#departure congestion
X_train, X_test = EXTRA_FT.add_departure_congestion(X_train, X_test, 30, "normalized")

#new numerical features
new_numerical_features = [col for col in X_train.columns if col not in original_cols]

#=========== encoding ===========
#ordinal encoder
ord_enc = OrdinalEncoder(categories=[['Very Short Haul', 'Short Haul', 'Medium Haul', 'Long Haul']])

#onehot encoder
oh_enc = OneHotEncoder(handle_unknown='ignore')

#standard scaler
ss_enc = StandardScaler()

#column transformer
col_tran = ColumnTransformer([("nominal", oh_enc, CONFIG.NOMINAL_FEATURES),
                              ("numerical", ss_enc, CONFIG.NUMERICAL_FEATURES + new_numerical_features),
                              ("ordinal", ord_enc, CONFIG.ORDINAL_FEATURES)])

#fitting and transforming
X_train_tran = col_tran.fit_transform(X_train)

X_test_tran = col_tran.transform(X_test)

# %%
#=========== models run ===========
#logistic regression
with mlflow.start_run(run_name="Logistic Regression - Features V8"):
    model = MODEL_LR.train(X_train_tran, y_train)

    y_pred = model.predict(X_test_tran)

    mlflow.set_tag("feature_set", "v8_normalized_congestion")

    mlflow.log_param("model", MODEL_LR.MODEL_NAME)
    mlflow.log_param("max_iter", MODEL_LR.MAX_ITER)
    mlflow.log_param("class_weight", MODEL_LR.CLASS_WEIGHT)
    mlflow.log_param("solver", MODEL_LR.SOLVER)

    EVAL.model_metrics(y_test, y_pred)
    
    mlflow.sklearn.log_model(model, "model")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(model.n_iter_)

#random forest
with mlflow.start_run(run_name="Random Forest - Features V8"):
    model = MODEL_RF.train(X_train_tran, y_train, CONFIG.RANDOM_STATE)

    y_pred = model.predict(X_test_tran)

    mlflow.set_tag("feature_set", "v8_normalized_congestion")

    mlflow.log_param("model", MODEL_RF.MODEL_NAME)
    mlflow.log_param("n_estimatores", MODEL_RF.N_ESTIMATORS)
    mlflow.log_param("max_depth", MODEL_RF.MAX_DEPTH)
    mlflow.log_param("class_weight", MODEL_RF.CLASS_WEIGHT)
    mlflow.log_param("min_samples_leaf", MODEL_RF.MIN_LEAF)
    mlflow.log_param("min_samples_split ", MODEL_RF.MIN_SPLIT)

    EVAL.model_metrics(y_test, y_pred)
    
    mlflow.sklearn.log_model(model, "model")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#xgboost
unbalanced_weight = MODEL_XGB.unbalanced_classes(y_train)

with mlflow.start_run(run_name="XGBoost - Features V8"):
    model = MODEL_XGB.train(X_train_tran, y_train, unbalanced_weight, CONFIG.RANDOM_STATE)

    y_pred = model.predict(X_test_tran)

    mlflow.set_tag("feature_set", "v8_normalized_congestion")

    mlflow.log_param("model", MODEL_XGB.MODEL_NAME)
    mlflow.log_param("n_estimatores", MODEL_XGB.N_ESTIMATORS)
    mlflow.log_param("max_depth", MODEL_XGB.MAX_DEPTH)
    mlflow.log_param("learning_rate", MODEL_XGB.LEARNING_RATE)
    mlflow.log_param("scale_pos_weight", unbalanced_weight)
    mlflow.log_param("eval_metric ", MODEL_XGB.EVAL_METRIC)

    EVAL.model_metrics(y_test, y_pred)
    
    mlflow.sklearn.log_model(model, "model")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#lightgbm
with mlflow.start_run(run_name="LightGBM - Features V8"):
    model = MODEL_LGBM.train(X_train_tran, y_train, CONFIG.RANDOM_STATE)

    y_pred = model.predict(X_test_tran)

    mlflow.set_tag("feature_set", "v8_normalized_congestion")

    mlflow.log_param("model", MODEL_LGBM.MODEL_NAME)
    mlflow.log_param("max_depth", MODEL_LGBM.MAX_DEPTH)
    mlflow.log_param("num_iterations", MODEL_LGBM.N_ITER)
    mlflow.log_param("learning_rate", MODEL_LGBM.LEARNING_RATE)
    mlflow.log_param("class_weight", MODEL_LGBM.CLASS_WEIGHT)

    EVAL.model_metrics(y_test, y_pred)
    
    mlflow.sklearn.log_model(model, "model")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))