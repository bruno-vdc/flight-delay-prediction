# =========== libraries ===========
from sklearn.linear_model import LogisticRegression

# =========== variables and functions ===========
MODEL_NAME = 'Logistic Regression'

def train_logistic(X_train, y_train):

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    return model