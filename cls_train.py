from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  

# Train the model
results = model.train(data="./hand-keypoints.yaml", epochs=100, imgsz=640)