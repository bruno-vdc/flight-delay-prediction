# =========== libraries ===========
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import config as CONFIG

# =========== functions ===========
def _time_to_minutes(time_str: str) -> int:
    hour, minute = map(int, time_str.split(":"))

    return hour * 60 + minute

def _get_season(month: int) -> str:
    seasons = {1: "Winter",
               2: "Winter",
               3: "Spring",
               4: "Spring",
               5: "Spring",
               6: "Summer",
               7: "Summer",
               8: "Summer",
               9: "Fall",
               10: "Fall",
               11: "Fall",
               12: "Winter"}

    return seasons[month]

def _prepare_dataframe(request, reference_data):
    flight_date = pd.to_datetime(request.FL_DATE)

    df = pd.DataFrame([{"AIRLINE_CODE": request.AIRLINE_CODE,
                        "ORIGIN": request.ORIGIN,
                        "DEST": request.DEST,
                        "year": flight_date.year,
                        "month": flight_date.month,
                        "day_of_week": flight_date.day_name(),
                        "season": _get_season(flight_date.month),
                        "is_holiday": False,
                        "crs_departure_minutes": _time_to_minutes(request.CRS_DEP_TIME),
                        "crs_arrival_minutes": _time_to_minutes(request.CRS_ARR_TIME)}])

    lookup = (reference_data.drop_duplicates(["ORIGIN", "DEST"]).set_index(["ORIGIN", "DEST"]))

    route = lookup.loc[(request.ORIGIN, request.DEST)]

    df["DISTANCE"] = route["DISTANCE"]
    df["CRS_ELAPSED_TIME"] = route["CRS_ELAPSED_TIME"]

    df["origin_state"] = route["origin_state"]
    df["origin_division"] = route["origin_division"]
    df["origin_contiguous"] = route["origin_contiguous"]

    df["dest_state"] = route["dest_state"]
    df["dest_division"] = route["dest_division"]
    df["dest_contiguous"] = route["dest_contiguous"]

    df["distance_category"] = route["distance_category"]

    return df

def predict(request, pipeline):
    model = pipeline["model"]
    transformer = pipeline["transformer"]
    reference_data = pipeline["reference_data"]

    df = _prepare_dataframe(request, reference_data)

    #calculate extra features
    route_mean = (reference_data[(reference_data["ORIGIN"]==request.ORIGIN)&(reference_data["DEST"]==request.DEST)]["route_delay_rate"].mean())
    origin_mean = (reference_data[reference_data["ORIGIN"]==request.ORIGIN]["origin_delay_rate"].mean())
    dest_mean = (reference_data[reference_data["DEST"]==request.DEST]["dest_delay_rate"].mean())
    airline_mean = (reference_data[reference_data["AIRLINE_CODE"]==request.AIRLINE_CODE]["airline_delay_rate"].mean())
    congestion = (reference_data[reference_data["crs_departure_minutes"]==df.loc[0, "crs_departure_minutes"]]["normalized_departure_congestion"].mean())

    #add extra features
    df["route_delay_rate"] = route_mean
    df["origin_delay_rate"] = origin_mean
    df["dest_delay_rate"] = dest_mean
    df["airline_delay_rate"] = airline_mean
    df["normalized_departure_congestion"] = congestion

    X = transformer.transform(df)

    probability = float(model.predict_proba(X)[0, 1])

    if probability < 0.30:
        risk = "Low"
    elif probability < 0.60:
        risk = "Medium"
    elif probability < 0.80:
        risk = "High"
    else:
        risk = "Very High"

    return {"delay_probability": round(probability * 100, 2),
            "risk": risk,
            "prepared_data": df}