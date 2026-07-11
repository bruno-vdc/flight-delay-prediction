# =========== libraries ===========
import pandas as pd

# =========== functions ===========
def generate_insights(df, reference_data, latest_training_year):

    #ingishts generation
    insights = []

    row = df.iloc[0]

    #airport congestion
    if row["normalized_departure_congestion"]>=reference_data["normalized_departure_congestion"].quantile(0.75):
        insights.append("The scheduled departure time is associated with a high level of airport congestion.")

    #route
    if row["route_delay_rate"]>=reference_data["route_delay_rate"].quantile(0.75):
        insights.append("This route has a historically high delay rate.")

    #airline
    if row["airline_delay_rate"]>=reference_data["airline_delay_rate"].quantile(0.75):
        insights.append("This airline has a higher historical delay rate.")

    #distance
    if row["DISTANCE"]>reference_data["DISTANCE"].quantile(0.75):
        insights.append("Longer-haul routes historically present greater operational complexity.")

    #elapsed time
    if row["CRS_ELAPSED_TIME"]>reference_data["CRS_ELAPSED_TIME"].quantile(0.75):
        insights.append("This flight has a longer-than-average scheduled elapsed duration.")

    #airports
    if row["origin_delay_rate"]>=reference_data["origin_delay_rate"].quantile(0.75):
        insights.append("Historical data indicate that flights departing from this airport experience above-average delays.")

    if row["dest_delay_rate"]>=reference_data["dest_delay_rate"].quantile(0.75):
        insights.append("Historical data indicate that flights arriving at this airport experience above-average delays.")

    #season
    if row["season"]=="Summer":
        insights.append("Summer is traditionally the busiest season for air traffic.")

    #fallback
    if len(insights)==0:
        insights.append("No significant historical risk factors were identified.")

    #warming message
    warning = None

    if row["year"]>latest_training_year:
        warning = (f"Prediction generated for a year outside the training "
                   f"period (2019-{latest_training_year}). Interpret the result with caution.")


    return {"insights": insights[:3],
            "warning": warning}