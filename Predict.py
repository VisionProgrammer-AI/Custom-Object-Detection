from ultralytics import YOLO


model = YOLO('runs/detect/train/weights/best.pt')

# Run inference
results = model("data/images/val/19.jpg",save=True)