# =========== libraries ===========
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#setting pandas options
pd.set_option('display.max_columns', None)   #columns
pd.set_option('display.max_rows', 10)        #rows

# =========== paths and files ===========
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / 'data'

df_flights = pd.read_csv(DATA_DIR / 'flights_sample_3m.csv')

# =========== data exploration ===========
# =========== adding useful columns ===========
df_flights['origin_state'] = df_flights['ORIGIN_CITY'].str[-2:]
df_flights['dest_state'] = df_flights['DEST_CITY'].str[-2:]

#origin and destination states lists
origin_states = sorted(df_flights['origin_state'].unique())
dest_states = sorted(df_flights['dest_state'].unique())

#U.S. states + D.C. list
us_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
             "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
             "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
             "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
             "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
             "DC"]
# =========== exploring data ===========
#basic info
# %%
df_flights.head()

# %%
df_flights.describe()

#checking for geographic coverage
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

# %%
#arrival delay
#plotting
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(17, 5))

#delay by airline
df_temp = df_flights.query('ARR_DELAY>14')['AIRLINE'].value_counts().to_frame(name="frequency").reset_index()
df_temp = df_temp.sort_values(by="frequency", ascending=False).reset_index(drop=True)

ax[0].bar(df_temp["AIRLINE"], df_temp["frequency"])
ax[0].set_title("Arrival Delay by Airline")
ax[0].set_xlabel("Airlines")
ax[0].set_ylabel("Frequency")
ax[0].tick_params(axis="x", labelrotation=80)

#delay by destination state
df_temp = df_flights.query('ARR_DELAY>14')['dest_state'].value_counts().to_frame(name="frequency").reset_index()
df_temp = df_temp.sort_values(by="frequency", ascending=False).reset_index(drop=True)

ax[1].bar(df_temp["dest_state"], df_temp["frequency"])
ax[1].set_title("Arrival Delay by Destination State")
ax[1].set_xlabel("Destination State")
ax[1].set_ylabel("Frequency")
ax[1].tick_params(axis="x", labelrotation=80)

# Improve spacing
plt.tight_layout()

# Show figure
plt.show()

# %%
df_temp = df_flights.query('ARR_DELAY>14')['AIRLINE'].value_counts().to_frame(name="frequency").reset_index()
df_temp = df_temp.sort_values(by="frequency", ascending=False).reset_index(drop=True)
