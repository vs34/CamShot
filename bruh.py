from ultralytics import YOLO



# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")


results = model.predict(source="0",show=True,classes=[0])