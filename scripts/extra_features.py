# =========== libraries ===========
import pandas as pd

# =========== variables and functions ===========
MIN_FLIGHTS = 10

def add_route_delay_rate(X_train, X_test, y_train, target_name, min_route_flights):

    #training dataframe
    train = X_train.copy()
    train[target_name] = y_train.values

    #global delay rate
    global_delay_rate = train[target_name].mean()

    #origin statistics
    origin_delay = (train.groupby("ORIGIN")[target_name].mean())

    #route statistics
    route_stats = train.groupby(["ORIGIN", "DEST"])[target_name].agg(["mean", "count"]).reset_index()
    route_stats.rename(columns={"mean": "route_delay_rate", "count": "route_flights"}, inplace=True)

    #fallback for rare routes
    route_stats["route_delay_rate"] = route_stats.apply(
        lambda row:
            row["route_delay_rate"]
            if row["route_flights"] >= min_route_flights
            else origin_delay[row["ORIGIN"]],
        axis=1)

    #create lookup dictionary
    route_delay_lookup  = (route_stats.set_index(["ORIGIN", "DEST"])["route_delay_rate"])

    #apply to train
    X_train = X_train.copy()

    X_train = X_train.merge(route_delay_lookup .rename("route_delay_rate"), on=["ORIGIN", "DEST"], how="left")

    #apply to test
    X_test = X_test.copy()

    X_test = X_test.merge(route_delay_lookup .rename("route_delay_rate"), on=["ORIGIN", "DEST"], how="left")

    #fallback for unseen routes
    X_train["route_delay_rate"] = (X_train["route_delay_rate"].fillna(X_train["ORIGIN"].map(origin_delay)).fillna(global_delay_rate))

    X_test["route_delay_rate"] = (X_test["route_delay_rate"].fillna(X_test["ORIGIN"].map(origin_delay)).fillna(global_delay_rate))

    return X_train, X_test