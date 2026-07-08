FEATURED_DATASET_PATH = 'data/flights_featured.csv'

TARGET = "is_delayed"
PREDICTOR_LIST = ['AIRLINE_CODE', 'ORIGIN', 'DEST', 'CRS_ELAPSED_TIME', 'DISTANCE', 'origin_state',
                  'origin_division', 'origin_contiguous', 'dest_state', 'dest_division', 'dest_contiguous',
                  'distance_category', 'crs_departure_minutes','crs_arrival_minutes', 'year', 'month',
                  'day_of_week', 'is_holiday', 'season']


TEST_SIZE = 0.2
RANDOM_STATE = 42