# =========== libraries ===========
from sklearn.ensemble import RandomForestClassifier

# =========== variables and functions ===========
MODEL_NAME = 'Random Forest'
N_ESTIMATORS = 50
MAX_DEPTH = 10

def train(X_train, y_train, ran_st):

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, class_weight="balanced",
                                   n_jobs=-1, random_state=ran_st)
    model.fit(X_train, y_train)

    return model