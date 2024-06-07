from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
#model = YOLO("yolov8n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="data.yaml", epochs=21)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("h.jpg",show=True)

#print(results)