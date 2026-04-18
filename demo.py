from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("images.jpeg", save=True)