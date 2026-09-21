#!/usr/bin/env python3
"""Download and export YOLOv8 model for traffic sign detection to ONNX."""
import urllib.request
import os

# Download a pre-trained YOLOv8 model for traffic signs
# Using YOLOv8n (nano) for speed - can be swapped for a traffic-sign specific model
model_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
model_path = "models/yolov8n.onnx"

os.makedirs("models", exist_ok=True)

print(f"Downloading YOLOv8n ONNX model to {model_path}...")
urllib.request.urlretrieve(model_url, model_path)
print(f"Downloaded {os.path.getsize(model_path)} bytes")

# The model detects 80 COCO classes. Traffic signs include:
# - stop sign (class 11)
# - traffic light (class 9) 
# - etc.
# For custom traffic sign detection, we'd need a custom-trained model
print("Note: This is general YOLOv8n (COCO classes).")
print("For traffic sign detection, you need a custom-trained model.")
print("Classes of interest: stop sign=11, traffic light=9")