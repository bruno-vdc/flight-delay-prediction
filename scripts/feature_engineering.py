# %%
# =========== libraries ===========
import pandas as pd
import numpy as np
from pathlib import Path
import holidays

#setting pandas options
pd.set_option('display.max_columns', None) #columns
pd.set_option('display.max_rows', 10)      #rows

# =========== paths and files ===========
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / 'data'

df_flights_proc = pd.read_csv(DATA_DIR / 'flights_sample_proc.csv')

#=========== restoring data types ===========
df_flights_proc['FL_DATE'] = pd.to_datetime(df_flights_proc['FL_DATE'], format='%Y-%m-%d')
df_flights_proc["CRS_DEP_TIME"] = pd.to_datetime(df_flights_proc["CRS_DEP_TIME"])
df_flights_proc["CRS_ARR_TIME"] = pd.to_datetime(df_flights_proc["CRS_ARR_TIME"])

# =========== lists and dictionaries ===========
#U.S.A. states and territories division
list_pacific = ['AK', 'WA', 'OR', 'CA', 'HI']
list_mountain = ['MT', 'ID', 'WY', 'NV', 'UT', 'CO', 'AZ', 'NM']
list_west_cnor = ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO']
list_west_csou = ['OK', 'AR', 'LA', 'TX']
list_east_cnor = ['WI', 'IL', 'MI', 'IN', 'OH']
list_east_csou = ['KY', 'TN', 'MS', 'AL']
list_mid_atlantic = ['PA', 'NY', 'NJ']
list_sou_atlantic = ['WV', 'VA', 'NC', 'SC', 'GA', 'FL', 'DC', 'MD', 'DE']
list_new_england = ['ME', 'NH', 'VT', 'MA', 'RI', 'CT']
list_caribbean = ['PR', 'VI']

#divisions dictionary
dict_divisions = {tuple(list_pacific): 'Pacific',
                  tuple(list_mountain): 'Mountain',
                  tuple(list_west_cnor): 'West North Central',
                  tuple(list_west_csou): 'West South Central',
                  tuple(list_east_cnor): 'East North Central',
                  tuple(list_east_csou): 'East South Central',
                  tuple(list_mid_atlantic): 'Middle Atlantic',
                  tuple(list_sou_atlantic): 'South Atlantic',
                  tuple(list_new_england): 'New England',
                  tuple(list_caribbean): 'Caribbean Islands'}

division_map = {}
for states, division in dict_divisions.items():
    for state in states:
        division_map[state] = division

#non-contiguous states and territories
list_non_contiguous = list_caribbean + ['AK', 'HI']

#distance categories
#kilometers per mile converter
kpm_conv = 1.609344
bins = [0, 500, 1500, 4000, float('inf')]
labels = ['Very Short Haul', 'Short Haul', 'Medium Haul', 'Long Haul']

#federal holidays
usa_holidays = holidays.US()

# =========== features ===========
flights_featured = df_flights_proc.copy()

#creating the target variable
flights_featured['is_delayed'] = np.where(flights_featured['ARR_DELAY']>=15, 1, 0)

#geographical features
flights_featured['origin_state'] = flights_featured['ORIGIN_CITY'].str[-2:]
flights_featured['origin_division'] = flights_featured['origin_state'].map(division_map)
flights_featured['origin_contiguous'] = np.where(flights_featured['origin_state'].isin(list_non_contiguous), 1, 0)

flights_featured['dest_state'] = flights_featured['DEST_CITY'].str[-2:]
flights_featured['dest_division'] = flights_featured['dest_state'].map(division_map)
flights_featured['dest_contiguous'] = np.where(flights_featured['dest_state'].isin(list_non_contiguous), 1, 0)

#distance categories feature
flights_featured['distance_category'] = pd.cut(flights_featured['DISTANCE']*kpm_conv, bins=bins, labels=labels, right=False)

#time features
flights_featured["crs_departure_minutes"] = ((flights_featured["CRS_DEP_TIME"].dt.hour*60)+flights_featured["CRS_DEP_TIME"].dt.minute)
flights_featured["crs_arrival_minutes"] = ((flights_featured["CRS_ARR_TIME"].dt.hour*60)+flights_featured["CRS_ARR_TIME"].dt.minute)

#date features
flights_featured["year"] = pd.to_datetime(flights_featured["FL_DATE"]).dt.year
flights_featured["month"] = pd.to_datetime(flights_featured["FL_DATE"]).dt.month
flights_featured["day_of_week"] = pd.to_datetime(flights_featured["FL_DATE"]).dt.day_name()

#holiday flag
flights_featured["is_holiday"] = (flights_featured["FL_DATE"].dt.date.isin(usa_holidays).astype(int))

#seasonal periods
periods = [(flights_featured["month"].isin([12, 1, 2])),
           (flights_featured["month"].isin([3, 4, 5])),
           (flights_featured["month"].isin([6, 7, 8])),
           (flights_featured["month"].isin([9, 10, 11]))]

seasons = ['Winter', 'Spring', 'Summer', 'Fall']

flights_featured['season'] = np.select(periods, seasons, default=None)

# %%
#checking featured dataset
flights_featured.head()

# =========== saving the final DF ===========
flights_featured.to_csv(DATA_DIR / 'flights_featured.csv', index=False)