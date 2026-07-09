# =========== libraries ===========
from sklearn.ensemble import RandomForestClassifier

# =========== variables and functions ===========
MODEL_NAME = 'Random Forest'

def train_forest(X_train, y_train, ran_st):

    model = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight="balanced",
                                   n_jobs=-1, random_state=ran_st)
    model.fit(X_train, y_train)

    return model