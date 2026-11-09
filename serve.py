from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import PredictiveMaintenanceModel


predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = PredictiveMaintenanceModel()
    yield
    predictor = None


app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure prediction using HistGradientBoosting",
    version="1.0.0",
    lifespan=lifespan,
)


class EquipmentInput(BaseModel):
    type: Literal["L", "M", "H"]
    air_temperature: float = Field(ge=0)
    process_temperature: float = Field(ge=0)
    rotational_speed: int = Field(gt=0)
    torque: float = Field(ge=0)
    tool_wear: int = Field(ge=0)


class PredictionOutput(BaseModel):
    machine_failure: int
    failure_probability: float
    threshold: float
    status: str


@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict_failure(equipment: EquipmentInput):
    equipment_data = {
        "Type": equipment.type,
        "Air temperature [K]": equipment.air_temperature,
        "Process temperature [K]": equipment.process_temperature,
        "Rotational speed [rpm]": equipment.rotational_speed,
        "Torque [Nm]": equipment.torque,
        "Tool wear [min]": equipment.tool_wear,
    }

    return predictor.predict(equipment_data)