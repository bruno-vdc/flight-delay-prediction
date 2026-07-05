# =========== libraries ===========
# %%
import pandas as pd
import numpy as np
from pathlib import Path

#setting pandas options
pd.set_option('display.max_columns', None) #columns
pd.set_option('display.max_rows', 10)      #rows

# =========== paths and files ===========
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / 'data'

df_flights = pd.read_csv(DATA_DIR / 'flights_sample_3m.csv')

df_flights.head(10)

# %%
#=========== removing cancelled flights ===========
df_flights_proc = df_flights.query('CANCELLED==0').reset_index(drop=True)

#=========== dropping unnecessary columns and missing values ===========
df_flights_proc = df_flights_proc.drop(columns=['CANCELLED', 'CANCELLATION_CODE', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
                                                'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT'])

miss_val_cols = ['ARR_DELAY', 'ELAPSED_TIME', 'AIR_TIME', 'TAXI_IN', 'WHEELS_ON', 'ARR_TIME', 'WHEELS_OFF', 'TAXI_OUT',
                 'DEP_DELAY', 'DEP_TIME', 'CRS_ELAPSED_TIME']

for col in miss_val_cols:
    query_text = f'{col}=={col}'
    df_flights_proc = df_flights_proc.query(query_text).reset_index(drop=True)

# =========== transforming date and time columns ===========
df_flights_proc['FL_DATE'] = pd.to_datetime(df_flights_proc['FL_DATE'], format='%Y-%m-%d')

time_cols = ['CRS_DEP_TIME', 'DEP_TIME', 'CRS_ARR_TIME', 'ARR_TIME']

for col in time_cols:
    df_flights_proc[col] = df_flights_proc[col].astype(int)
    df_flights_proc[col] = df_flights_proc[col].astype(str).str.zfill(4)
    df_flights_proc[col] = np.where(df_flights_proc[col]=='2400', '0000', df_flights_proc[col])
    df_flights_proc[col] = pd.to_datetime(df_flights_proc[col], format='%H%M')
