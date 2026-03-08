from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolo26n-pose.pt')

# Export the model to TensorRT format
model.export(format='engine', device='0', half=True)  # half=True for FP16 optimization