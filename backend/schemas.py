from typing import Literal, List

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    video_id: str = Field(..., description="Uploaded video filename")
    label: Literal["REAL", "FAKE"] = Field(..., description="Predicted label")
    fake_probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    num_frames_used: int
    inference_time_ms: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "video_id": "01_02__exit_phone_room__YVGY8LOK.mp4",
                "label": "FAKE",
                "fake_probability": 0.9317,
                "threshold": 0.5,
                "num_frames_used": 15,
                "inference_time_ms": 412.7,
            }
        }
    }


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_name: str
    device: str
    checkpoint: str


class ModelsResponse(BaseModel):
    """Response schema for listing available models."""

    models: List[str] = Field(..., description="List of available model names")
    default: str = Field(..., description="Default model name")


class ErrorResponse(BaseModel):
    detail: str
