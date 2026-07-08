# %%
# =========== libraries ===========
import pandas as pd
from pathlib import Path
from config import FEATURED_DATASET_PATH, TARGET, PREDICTOR_LIST, TEST_SIZE, RANDOM_STATE
from models.logistic_regression import train_logistic
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#setting pandas options
pd.set_option('display.max_columns', None) #columns
pd.set_option('display.max_rows', 10)      #rows

# =========== paths and files ===========
PROJECT_ROOT = Path(__file__).resolve().parent.parent

flights_featured = pd.read_csv(PROJECT_ROOT/FEATURED_DATASET_PATH)

#=========== restoring data types ===========
flights_featured['FL_DATE'] = pd.to_datetime(flights_featured['FL_DATE'], format='%Y-%m-%d')
flights_featured["CRS_DEP_TIME"] = pd.to_datetime(flights_featured["CRS_DEP_TIME"])
flights_featured["CRS_ARR_TIME"] = pd.to_datetime(flights_featured["CRS_ARR_TIME"])

#=========== model fit ===========
X = flights_featured[PREDICTOR_LIST]
y = flights_featured[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

#=========== encoding ===========


#=========== logistic regression ===========
model_log_reg = train_logistic(X_train, y_train)

y_log_reg_pred = model_log_reg.predict(X_test)

print(classification_report(y_test, y_log_reg_pred))