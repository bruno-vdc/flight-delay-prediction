FEATURED_DATASET_PATH = 'data/flights_featured.csv'

TARGET = "is_delayed"
NOMINAL_FEATURES = ['AIRLINE_CODE', 'ORIGIN', 'DEST', 'origin_state', 'origin_division', 'origin_contiguous', 'dest_state', 'dest_division',
                    'dest_contiguous', 'day_of_week', 'season']
ORDINAL_FEATURES = ['distance_category']
NUMERICAL_FEATURES = ['CRS_ELAPSED_TIME', 'DISTANCE', 'crs_departure_sin', 'crs_departure_cos', 'crs_arrival_sin', 'crs_arrival_cos',
                      'crs_departure_minutes', 'crs_arrival_minutes', 'year', 'month', 'is_holiday']
PREDICTOR_LIST = NOMINAL_FEATURES+ORDINAL_FEATURES+NUMERICAL_FEATURES

TEST_SIZE = 0.2
RANDOM_STATE = 42