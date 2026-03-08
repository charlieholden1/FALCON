from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolo26n-pose.pt')

# Export the model to TensorRT format
# simplify=False is critical on Jetson to avoid installing onnxsim (which requires building cmake)
model.export(format='engine', device='0', half=True, simplify=False)