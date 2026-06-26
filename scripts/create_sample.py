# =========== libraries ===========
import pandas as pd
from pathlib import Path

# =========== paths and files ===========
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

FULL_DATASET = DATA_DIR / "flights_sample_3m.csv" #downloaded from Kaggle
SAMPLE_DATASET = DATA_DIR / "sample_flights.csv"  #10k rows sample dataset

df = pd.read_csv(FULL_DATASET)

# =========== create reproducible sample ===========
df_sample = df.sample(n=10_000, random_state=42)

df_sample.to_csv(SAMPLE_DATASET, index=False)
