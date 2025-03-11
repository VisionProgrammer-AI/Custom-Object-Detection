from ultralytics import YOLO

# Load a COCO-pretrained YOLO12n model
model = YOLO("yolo12n.pt")

# Train the model on the custom dataset for 100 epochs
results = model.train(data="cone.yaml", epochs=100, imgsz=640)
