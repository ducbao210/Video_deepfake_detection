import shutil
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.dependencies import get_model_bundle
from backend.schemas import HealthResponse, PredictionResponse
from scripts.inference import extract_frames_memory

ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model_bundle()
    yield


app = FastAPI(
    title="Video Deepfake Detection API",
    version="1.0.0",
    description="Video deepfake detection using ConvNeXt / BiLSTM / Video Swin",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    bundle = get_model_bundle()
    return HealthResponse(
        status="ok",
        model_name=bundle.cfg.model.name,
        device=str(bundle.device),
        checkpoint=str(bundle.cfg.inference.checkpoint),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format: '{suffix}'. "
            f"Supported formats: {sorted(ALLOWED_SUFFIXES)}",
        )

    bundle = get_model_bundle()
    cfg = bundle.cfg

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            written = 0
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File size exceeds the 200 MB limit.")
                tmp.write(chunk)

        started = time.perf_counter()

        raw_frames = extract_frames_memory(
            tmp_path, cfg.preprocessing.frame_count, cfg.preprocessing.image_size
        )
        tensor = torch.stack([bundle.transform(f) for f in raw_frames])
        tensor = tensor.unsqueeze(0).to(bundle.device)

        with torch.inference_mode():
            logits = bundle.model(tensor)
            prob = torch.softmax(logits, dim=1)[:, 1].item()

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        threshold = float(cfg.inference.threshold)

        return PredictionResponse(
            video_id=file.filename or tmp_path.name,
            label="FAKE" if prob >= threshold else "REAL",
            fake_probability=round(prob, 4),
            threshold=threshold,
            num_frames_used=len(raw_frames),
            inference_time_ms=round(elapsed_ms, 1),
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, f"Invalid video: {e}")
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        await file.close()
