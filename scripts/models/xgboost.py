# =========== libraries ===========
from xgboost import XGBClassifier

# =========== variables and functions ===========
MODEL_NAME = 'XGBoost'
N_ESTIMATORS = 100
MAX_DEPTH = 10
LEARNING_RATE = 0.5
EVAL_METRIC = "logloss"

def unbalanced_classes(y_train):
    neg = (y_train==0).sum()
    pos = (y_train==1).sum()

    unbalanced_weight = neg/pos

    return unbalanced_weight

def train(X_train, y_train, unbalanced_weight, ran_st):

    model = XGBClassifier(scale_pos_weight=unbalanced_weight, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                          learning_rate=LEARNING_RATE, random_state=ran_st, eval_metric=EVAL_METRIC, device="gpu")
    
    model.fit(X_train, y_train)

    return model