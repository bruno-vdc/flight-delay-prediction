# =========== libraries ===========
import pandas as pd

# =========== variables and functions ===========
MIN_FLIGHTS = 10

#route delay rates
def add_route_delay_rate(X_train, X_test, y_train, target_name, min_route_flights):

    #training dataframe
    train = X_train.copy()
    train[target_name] = y_train.values

    #global delay rate
    global_delay_rate = train[target_name].mean()

    #origin statistics
    origin_delay = train.groupby("ORIGIN")[target_name].mean()

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

#general delay rate
def add_delay_rate_feature(X_train, X_test, y_train, target_name, group_column, feature_name):
    
    #training dataframe
    train = X_train.copy()

    train[target_name] = y_train.values

    #global delay rate
    global_delay_rate = train[target_name].mean()

    #group statistics
    delay_rate = (train.groupby(group_column)[target_name].mean().rename(feature_name))

    #apply to train
    X_train = X_train.copy()

    X_train = X_train.merge(delay_rate, on=group_column, how="left")

    #apply to test
    X_test = X_test.copy()

    X_test = X_test.merge(delay_rate, on=group_column, how="left")

    #fallback for unseen categories
    X_train[feature_name] = (X_train[feature_name].fillna(global_delay_rate))

    X_test[feature_name] = (X_test[feature_name].fillna(global_delay_rate))

    return X_train, X_test

#airport congestion
def add_departure_congestion(X_train, X_test, time_slot, method):
    """
    Creates a congestion feature based on scheduled departures.

    Parameters
    ----------
    time_slot : int
        Size of the time window in minutes.
        Supported values: 30 or 60.

    method : str
        Congestion calculation.
        Supported values:
            - absolute
            - relative
            - normalized"""

    #parameter validation
    if time_slot not in [30, 60]:
        raise ValueError("time_slot must be either 30 or 60 minutes.")

    if method not in ["absolute", "relative", "normalized"]:
        raise ValueError("method must be 'absolute', 'relative' or 'normalized'.")

    #copy dataframes
    X_train = X_train.copy()
    X_test = X_test.copy()

    slot_col = f"departure_slot_{time_slot}"

    X_train[slot_col] = (X_train["crs_departure_minutes"]//time_slot)

    X_test[slot_col] = (X_test["crs_departure_minutes"]//time_slot)

    #historical congestion
    congestion = (X_train.groupby(["ORIGIN", "day_of_week", slot_col]).size().reset_index(name="departure_congestion"))

    feature_name = f"{method}_departure_congestion"

    #feature calculation
    if method == "absolute":
        congestion[feature_name] = (congestion["departure_congestion"])
    elif method == "relative":
        airport_mean = (congestion.groupby("ORIGIN")["departure_congestion"].mean().reset_index(name="airport_mean"))
        congestion = congestion.merge(airport_mean, on="ORIGIN", how="left")
        congestion[feature_name] = (congestion["departure_congestion"]/congestion["airport_mean"])
    else:
        airport_stats = (congestion.groupby("ORIGIN")["departure_congestion"].agg(["mean", "std"]).reset_index())
        congestion = congestion.merge(airport_stats, on="ORIGIN", how="left")
        congestion["std"] = (congestion["std"].replace(0, 1).fillna(1))
        congestion[feature_name] = ((congestion["departure_congestion"]-congestion["mean"])/congestion["std"])

    #keep only necessary columns
    congestion = congestion[["ORIGIN", "day_of_week", slot_col, feature_name]]

    #========== merge ==========
    X_train = X_train.merge(congestion, on=["ORIGIN", "day_of_week", slot_col], how="left")

    X_test = X_test.merge(congestion, on=["ORIGIN", "day_of_week", slot_col], how="left")

    #fallback
    global_value = congestion[feature_name].mean()

    X_train[feature_name] = (X_train[feature_name].fillna(global_value))

    X_test[feature_name] = (X_test[feature_name].fillna(global_value))

    #cleanup
    X_train.drop(columns=slot_col, inplace=True)
    X_test.drop(columns=slot_col, inplace=True)

    return X_train, X_test