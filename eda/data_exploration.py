# =========== libraries ===========
# %%
import pandas as pd
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

#saving the figure
plt.savefig(IMAGES_DIR / "missing_values.png", dpi=300, bbox_inches="tight")

#show figure
plt.show()

#correlation matrix
# %%
corr = df_flights[hm_cols].corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.savefig(IMAGES_DIR / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
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

#improve spacing
plt.tight_layout()

#saving the figure
plt.savefig(IMAGES_DIR / "delay_distributions.png", dpi=300, bbox_inches="tight")

#show figure
plt.show()

# %%
#airline frequency and delay rates
df_temp = percentage_dataframe(df_flights, "AIRLINE", "total_flights")

#plotting
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

#airlines frequency
ax[0].bar(df_temp["AIRLINE"], df_temp["total_flights"])
ax[0].set_title("Airlines Frequency")
ax[0].set_xlabel("Airlines")
ax[0].set_ylabel("Frequency")
ax[0].tick_params(axis="x", labelrotation=80)

#arrival delay distribution
df_temp = df_temp.sort_values(by="percentage_delayed", ascending=False).reset_index(drop=True)

ax[1].bar(df_temp["AIRLINE"], df_temp["percentage_delayed"])
ax[1].set_title("Arrival Delay Distribution")
ax[1].set_xlabel("Airlines")
ax[1].set_ylabel("Delay Rate (%)")
ax[1].tick_params(axis="x", labelrotation=80)

#adjust spacing
plt.tight_layout()

#saving the figure
plt.savefig(IMAGES_DIR / "airline_frequency_delay.png", dpi=300, bbox_inches="tight")

#show figure
plt.show()

# %%
#monthly demand and delay rates
df_temp = percentage_dataframe(df_flights, "month", "total_flights")

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#plotting
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

#monthly demand
ax[0].bar(df_temp["month"], df_temp["total_flights"])
ax[0].set_title("Monthly Demand")
ax[0].set_xlabel("Month")
ax[0].set_ylabel("Total Flights")
ax[0].set_xticks(range(1, 13))
ax[0].set_xticklabels(month_labels)

#delay rate by month
df_temp = df_temp.sort_values(by="percentage_delayed", ascending=False).reset_index(drop=True)

#monthly demand
ax[1].bar(df_temp["month"], df_temp["percentage_delayed"])
ax[1].set_title("Monthly Delay Rate")
ax[1].set_xlabel("Month")
ax[1].set_ylabel("Delay Rate (%)")
ax[1].set_xticks(range(1, 13))
ax[1].set_xticklabels(month_labels)

#adjust spacing
plt.tight_layout()

#saving the figure
plt.savefig(IMAGES_DIR / "monthly_delay_rates.png", dpi=300, bbox_inches="tight")

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
ax[0].tick_params(axis="x", labelrotation=90)

#delay by destination state
df_temp = percentage_dataframe(df_flights, "dest_state", "percentage_delayed")

ax[1].bar(df_temp["dest_state"], df_temp["percentage_delayed"])
ax[1].set_title("Arrival Delay by Destination State")
ax[1].set_xlabel("Destination State")
ax[1].set_ylabel("Delay Rate (%)")
ax[1].tick_params(axis="x", labelrotation=90)

# Improve spacing
plt.tight_layout()

#saving the figure
plt.savefig(IMAGES_DIR / "delay_by_states.png", dpi=300, bbox_inches="tight")

# Show figure
plt.show()