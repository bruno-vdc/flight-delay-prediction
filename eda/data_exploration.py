# =========== libraries ===========
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#setting pandas options
pd.set_option('display.max_columns', None) #columns
pd.set_option('display.max_rows', 10)      #rows

# =========== paths and files ===========
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / 'data'
IMAGES_DIR = PROJECT_ROOT / 'images'

df_flights = pd.read_csv(DATA_DIR / 'flights_sample_3m.csv')

# =========== functions ===========
def percentage_dataframe(df, column, sort_col):
    """
    function to calculate the percentage of delayed flights for a given column in the dataframe.
    """
    df_agg = df.groupby(column)[['FL_NUMBER']].count().reset_index().rename(columns={'FL_NUMBER': 'total_flights'})

    df_temp = df.query('ARR_DELAY>14').reset_index(drop=True)
    df_temp = df_temp.groupby(column)[['FL_NUMBER']].count().reset_index().rename(columns={'FL_NUMBER': 'delayed_flights'})
    df_temp = df_temp.merge(df_agg, on=column, how='left')
    df_temp['percentage_delayed'] = (df_temp['delayed_flights']/df_temp['total_flights'])*100
    df_temp = df_temp.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    return df_temp

# =========== data exploration ===========
# =========== creating useful columns and lists ===========
df_flights['origin_state'] = df_flights['ORIGIN_CITY'].str[-2:]
df_flights['dest_state'] = df_flights['DEST_CITY'].str[-2:]
df_flights["month"] = pd.to_datetime(df_flights["FL_DATE"]).dt.month

#origin and destination states lists
origin_states = sorted(df_flights['origin_state'].unique())
dest_states = sorted(df_flights['dest_state'].unique())

#U.S. states + D.C. list
us_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
             "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
             "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"]

#correlation matrix columns
hm_cols = ['month',  'TAXI_IN',  'ELAPSED_TIME', 'DELAY_DUE_CARRIER', 'DEP_DELAY', 'ARR_DELAY', 'DELAY_DUE_LATE_AIRCRAFT',
           'DELAY_DUE_SECURITY', 'CRS_ELAPSED_TIME', 'TAXI_OUT', 'AIR_TIME', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DISTANCE']

# =========== exploring data ===========
#basic info
# %%
df_flights.head()

# %%
df_flights.describe()

#checking for geographic coverage of the dataset
# %%
print(set(origin_states) - set(us_states))
print(set(dest_states) - set(us_states))
print(set(us_states) - set(origin_states))
print(set(us_states) - set(dest_states))

#null values
# %%
df_missing = pd.DataFrame({"column": df_flights.columns,
                           "missing_values": df_flights.isnull().sum().values,
                           "percentage": (df_flights.isnull().sum()/len(df_flights)*100).values})

df_missing = df_missing.query('missing_values>0').sort_values(by="missing_values", ascending=False).reset_index(drop=True)

#plotting missing values
plt.figure(figsize=(9, 5))

plt.bar(df_missing["column"], df_missing["missing_values"])

#customize
plt.title("Missing Values by Column")
plt.xlabel("Column")
plt.ylabel("Missing Values")
plt.xticks(rotation=45, ha="right")

#adjust spacing
plt.tight_layout()

#show figure
plt.show()

#correlation matrix
# %%
corr = df_flights[hm_cols].corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()

# %%
#most frequent airlines
df_freq_airlines = df_flights['AIRLINE'].value_counts().to_frame(name="frequency").reset_index()
df_freq_airlines = df_freq_airlines.sort_values(by="frequency", ascending=True).reset_index(drop=True)

plt.figure(figsize=(8, 5))

# Plot
plt.barh(df_freq_airlines["AIRLINE"], df_freq_airlines["frequency"])

# Customize
plt.title("Airlines Frequency")
plt.xlabel("Frequency")
plt.ylabel("Airlines")

# Adjust spacing
plt.tight_layout()

# Show figure
plt.show()

#most delayed airlines
df_temp = percentage_dataframe(df_flights, "AIRLINE", "percentage_delayed")

#plotting delay rates
plt.figure(figsize=(9, 5))

plt.bar(df_temp["AIRLINE"], df_temp["percentage_delayed"])

#customize
plt.title("Delay Rate by Airline")
plt.xlabel("Airlines")
plt.ylabel("Delay Rate (%)")
plt.xticks(rotation=45, ha="right")

#adjust spacing
plt.tight_layout()

#show figure
plt.show()

# %%
#delay time distribution
#plotting
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

#departure delay distribution
ax[0].hist(df_flights["DEP_DELAY"].dropna(), bins=50, edgecolor="black")
ax[0].set_title("Departure Delay Distribution")
ax[0].set_xlabel("Departure Delay (minutes)")
ax[0].set_ylabel("Frequency")

#arrival delay distribution
ax[1].hist(df_flights["ARR_DELAY"].dropna(), bins=50, edgecolor="black")
ax[1].set_title("Arrival Delay Distribution")
ax[1].set_xlabel("Arrival Delay (minutes)")
ax[1].set_ylabel("Frequency")

# Improve spacing
plt.tight_layout()

# Show figure
plt.show()

# %%
#most delayed months
df_temp = percentage_dataframe(df_flights, "month", "percentage_delayed")

#plotting delay rates
plt.figure(figsize=(9, 5))

plt.bar(df_temp["month"], df_temp["percentage_delayed"])

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#customize
plt.title("Delay Rate by Month")
plt.xlabel("Month")
plt.ylabel("Delay Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.xticks(range(1, 13), month_labels)

#adjust spacing
plt.tight_layout()

#show figure
plt.show()

#delay rate by origin and destination states
# %%
#plotting
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(17, 5))

#delay by origin state
df_temp = percentage_dataframe(df_flights, "origin_state", "percentage_delayed")

ax[0].bar(df_temp["origin_state"], df_temp["percentage_delayed"])
ax[0].set_title("Arrival Delay by Origin State")
ax[0].set_xlabel("Origin State")
ax[0].set_ylabel("Delay Rate (%)")
ax[0].tick_params(axis="x", labelrotation=60)

#delay by destination state
df_temp = percentage_dataframe(df_flights, "dest_state", "percentage_delayed")

ax[1].bar(df_temp["dest_state"], df_temp["percentage_delayed"])
ax[1].set_title("Arrival Delay by Destination State")
ax[1].set_xlabel("Destination State")
ax[1].set_ylabel("Delay Rate (%)")
ax[1].tick_params(axis="x", labelrotation=60)

# Improve spacing
plt.tight_layout()

# Show figure
plt.show()