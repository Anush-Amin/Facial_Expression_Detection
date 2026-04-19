import numpy as np
from fastapi import HTTPException
import cv2
import base64

def run_detection(cv_image: np.ndarray, model):
    # YOLO expects BGR or RGB; cv2 returns BGR
    results = model.predict(source=cv_image, conf=0.25, imgsz=640, verbose=False)
    if len(results) == 0:
        raise HTTPException(status_code=500, detail="Model inference failed.")
    r = results[0]

    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
            detections.append({
                "label": model.names.get(cls, str(cls)),
                "confidence": float(conf),
                "box": [float(x1), float(y1), float(x2), float(y2)],
            })

    annotated = r.plot() if hasattr(r, "plot") else cv_image
    if annotated is None:
        annotated = cv_image

    # Encode annotated image as base64 string for easy HTML display
    _, buffer = cv2.imencode('.jpg', annotated)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "detections": detections,
        "image_data": f"data:image/jpeg;base64,{image_base64}",
    }
