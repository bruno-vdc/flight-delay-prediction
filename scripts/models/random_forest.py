# =========== libraries ===========
from sklearn.ensemble import RandomForestClassifier

# =========== variables and functions ===========
MODEL_NAME = 'Random Forest'
N_ESTIMATORS = 100
MAX_DEPTH = 20
CLASS_WEIGHT = "balanced"
MIN_SPLIT = 20
MIN_LEAF = 10

def train(X_train, y_train, ran_st):

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, class_weight=CLASS_WEIGHT,
                                   n_jobs=-1, random_state=ran_st, oob_score=True, min_samples_split=MIN_SPLIT,
                                   min_samples_leaf=MIN_LEAF)
    model.fit(X_train, y_train)

    return model