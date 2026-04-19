from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from pathlib import Path
from src.helper import run_detection

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Facial Emotion Detection System")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Load the YOLO model
model = YOLO(model=str(BASE_DIR / "best.pt"))

class WebcamPayload(BaseModel):
    image: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = run_detection(image, model)
    return JSONResponse(result)

@app.post("/detect/webcam")
async def detect_webcam(payload: WebcamPayload):
    img_data = payload.image
    if img_data.startswith("data:"):
        img_data = img_data.split(",", 1)[1]

    try:
        decoded = base64.b64decode(img_data)
        nparr = np.frombuffer(decoded, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid webcam image: {e}")

    result = run_detection(image, model)
    return JSONResponse(result)

if __name__ == "__main__":
    import uvicorn
    host = "0.0.0.0"
    port = 8000
    print(f"Starting Facial Emotion Detection app on http://{host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=True)