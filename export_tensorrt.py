import os
import ctypes

# Force load libgomp before anything else to fix "cannot allocate memory in static TLS block"
try:
    ctypes.CDLL('/usr/lib/aarch64-linux-gnu/libgomp.so.1', mode=ctypes.RTLD_GLOBAL)
except OSError:
    print("Warning: Could not preload libgomp.so.1 - you might see TLS errors.")

from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolo26n-pose.pt')

# Export the model to TensorRT format
# simplify=False is critical on Jetson to avoid installing onnxsim (which requires building cmake)
model.export(format='engine', device='0', half=True, simplify=False)