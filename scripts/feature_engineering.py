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

# =========== features ===========
#creating the target variable
df_flights_proc['is_delayed'] = np.where(df_flights_proc['ARR_DELAY']>=15, 1, 0)
