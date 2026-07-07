# %%
# =========== libraries ===========
import pandas as pd
import numpy as np
from pathlib import Path

#setting pandas options
pd.set_option('display.max_columns', None) #columns
pd.set_option('display.max_rows', 10)      #rows

# =========== paths and files ===========
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / 'data'

df_flights_proc = pd.read_csv(DATA_DIR / 'flights_sample_proc.csv')

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

flight_distance = [(df_flights_proc['DISTANCE']*kpm_conv < 500),
                   (df_flights_proc['DISTANCE']*kpm_conv >= 500) & (df_flights_proc['DISTANCE']*kpm_conv < 1500),
                   (df_flights_proc['DISTANCE']*kpm_conv >= 1500) & (df_flights_proc['DISTANCE']*kpm_conv < 4000),
                   (df_flights_proc['DISTANCE']*kpm_conv > 4000)]

distance_class = ['Very Short Haul', 'Short Haul', 'Medium Haul', 'Long Haul']

# %%
# =========== features ===========
#creating the target variable
df_flights_proc['is_delayed'] = np.where(df_flights_proc['ARR_DELAY']>=15, 1, 0)

#geographical features
df_flights_proc['origin_state'] = df_flights_proc['ORIGIN_CITY'].str[-2:]
df_flights_proc['origin_division'] = df_flights_proc['origin_state'].map(division_map)
df_flights_proc['origin_contiguous'] = np.where(df_flights_proc['origin_state'].isin(list_non_contiguous), 1, 0)

df_flights_proc['dest_state'] = df_flights_proc['DEST_CITY'].str[-2:]
df_flights_proc['dest_division'] = df_flights_proc['dest_state'].map(division_map)
df_flights_proc['dest_contiguous'] = np.where(df_flights_proc['dest_state'].isin(list_non_contiguous), 1, 0)

#distance categories feature
df_flights_proc['distance_category'] = np.select(flight_distance, distance_class)

#time and date features