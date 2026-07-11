# =========== libraries ===========
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import config as CONFIG
import extra_features as EXTRA_FT
import models.xgboost as MODEL_XGB

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def train_model():

    # =========== load dataset ===========
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    flights_featured = pd.read_csv(PROJECT_ROOT / CONFIG.FEATURED_DATASET_PATH)

    X = flights_featured[CONFIG.PREDICTOR_LIST].copy()
    y = flights_featured[CONFIG.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=CONFIG.TEST_SIZE, random_state=CONFIG.RANDOM_STATE)

    # =========== original columns ===========
    original_cols = X_train.columns.tolist()

    # =========== additional features ===========
    X_train, X_test = EXTRA_FT.add_route_delay_rate(X_train, X_test, y_train, CONFIG.TARGET, EXTRA_FT.MIN_FLIGHTS)

    X_train, X_test = EXTRA_FT.add_delay_rate_feature(X_train, X_test, y_train, CONFIG.TARGET, group_column="ORIGIN", feature_name="origin_delay_rate")

    X_train, X_test = EXTRA_FT.add_delay_rate_feature(X_train, X_test, y_train, CONFIG.TARGET, group_column="DEST", feature_name="dest_delay_rate")

    X_train, X_test = EXTRA_FT.add_delay_rate_feature(X_train, X_test, y_train, CONFIG.TARGET, group_column="AIRLINE_CODE", feature_name="airline_delay_rate")

    X_train, X_test = EXTRA_FT.add_departure_congestion( X_train, X_test, 30, "normalized")

    # =========== encoders ===========
    ord_enc = OrdinalEncoder(categories=[["Very Short Haul", "Short Haul", "Medium Haul", "Long Haul"]])

    oh_enc = OneHotEncoder(handle_unknown="ignore")

    ss_enc = StandardScaler()

    new_numerical_features = [col for col in X_train.columns if col not in original_cols]

    #column transformer
    col_tran = ColumnTransformer([("nominal", oh_enc, CONFIG.NOMINAL_FEATURES),
                                  ("numerical", ss_enc, CONFIG.NUMERICAL_FEATURES + new_numerical_features),
                                  ("ordinal", ord_enc, CONFIG.ORDINAL_FEATURES)])

    X_train_tran = col_tran.fit_transform(X_train)

    # =========== xgboost ===========
    unbalanced_weight = MODEL_XGB.unbalanced_classes(y_train)

    model = MODEL_XGB.train(X_train_tran, y_train, unbalanced_weight, CONFIG.RANDOM_STATE)

    return {"model": model,
            "transformer": col_tran,
            "training_columns": X_train.columns.tolist(),
            "predictor_list": CONFIG.PREDICTOR_LIST,
            "new_features": new_numerical_features,
            "reference_data": X_train.copy(),
            "y_train": y_train.copy(),
            "X_train": X_train.copy()}