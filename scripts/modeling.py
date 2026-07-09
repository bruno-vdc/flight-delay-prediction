# %%
# =========== libraries ===========
import pandas as pd
from pathlib import Path
from config import FEATURED_DATASET_PATH, TARGET, PREDICTOR_LIST, NOMINAL_FEATURES, ORDINAL_FEATURES, TEST_SIZE, RANDOM_STATE
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from models.logistic_regression import train_logistic
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#setting pandas options
pd.set_option("display.max_columns", None) #columns
pd.set_option("display.max_rows", 10)      #rows

# =========== paths and files ===========
PROJECT_ROOT = Path(__file__).resolve().parent.parent

flights_featured = pd.read_csv(PROJECT_ROOT/FEATURED_DATASET_PATH)

#=========== model fit ===========
X = flights_featured[PREDICTOR_LIST]
y = flights_featured[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

#=========== encoding ===========
#ordinal encoder
ord_enc = OrdinalEncoder(categories=[['Very Short Haul', 'Short Haul', 'Medium Haul', 'Long Haul']])

#onehot encoder
oh_enc = OneHotEncoder(handle_unknown='ignore')

#column transformer
col_tran = ColumnTransformer([("nominal", oh_enc, NOMINAL_FEATURES), ("ordinal", ord_enc, ORDINAL_FEATURES)], remainder='passthrough')

#fitting and transforming
X_train_tran = col_tran.fit_transform(X_train)

X_test_tran = col_tran.transform(X_test)

#=========== logistic regression ===========
model_log_reg = train_logistic(X_train_tran, y_train)

y_log_reg = model_log_reg.predict(X_test_tran)

print(classification_report(y_test, y_log_reg))