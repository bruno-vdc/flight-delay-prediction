# =========== libraries ===========
# %%
import pandas as pd
from pathlib import Path

#setting pandas options
pd.set_option('display.max_columns', None) #columns
pd.set_option('display.max_rows', 10)      #rows

# =========== paths and files ===========
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / 'data'

df_flights = pd.read_csv(DATA_DIR / 'flights_sample_3m.csv')

# %%
df_flights.head(10)
