# =========== libraries ===========
from pydantic import BaseModel
from typing import Optional

# =========== classes ===========
class FlightRequest(BaseModel):
    AIRLINE_CODE: str

    #airport codes
    ORIGIN: str
    DEST: str

    #yyyy-mm-dd flight date
    FL_DATE: str

    #HH:MM scheduled departure and arrival times
    CRS_DEP_TIME: str
    CRS_ARR_TIME: str

class PredictionResponse(BaseModel):
    delay_probability: float
    risk: str
    insights: list[str]
    warning: Optional[str]=None