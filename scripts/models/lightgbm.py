# =========== libraries ===========
from lightgbm import LGBMClassifier

# =========== variables and functions ===========
MODEL_NAME = 'LightGBM'
N_ITER = 100
MAX_DEPTH = 10
LEARNING_RATE = 0.1
CLASS_WEIGHT = "balanced"

def train(X_train, y_train, ran_st):

    model = LGBMClassifier(max_depth=MAX_DEPTH, num_iterations=N_ITER, class_weight=CLASS_WEIGHT,
                           learning_rate=LEARNING_RATE, random_state=ran_st, device="gpu")
    
    model.fit(X_train, y_train)

    return model