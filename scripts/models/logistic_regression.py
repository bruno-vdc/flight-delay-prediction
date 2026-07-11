# =========== libraries ===========
from sklearn.linear_model import LogisticRegression

# =========== variables and functions ===========
MODEL_NAME = 'Logistic Regression'
MAX_ITER = 20
CLASS_WEIGHT = "balanced"
SOLVER = "lbfgs"

def train(X_train, y_train):

    model = LogisticRegression(max_iter=MAX_ITER, class_weight=CLASS_WEIGHT, n_jobs=-1, solver=SOLVER)
    model.fit(X_train, y_train)

    return model