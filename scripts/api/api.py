# =========== libraries ===========
from fastapi import FastAPI, HTTPException

from train_api import train_model
from predictor import predict
from schemas import FlightRequest, PredictionResponse

# =========== application ===========
app = FastAPI(title="Flight Delay Prediction API",
              description="Flight delay prediction API for the USA",
              version="1.0.0")


# =========== load model ===========
print("Loading prediction pipeline...")

pipeline = train_model()

print("Pipeline loaded successfully!")

# =========== routes ===========
@app.get("/")
def home():
    return {"message": "Flight Delay Prediction API is running."}


@app.post("/predict", response_model=PredictionResponse)

def predict_delay(request: FlightRequest):

    try:
        prediction = predict(request, pipeline)

        return prediction

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))