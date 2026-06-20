from ultralytics import YOLO

# # Load a pre-trained model
# model = YOLO('yolov8n.pt')

# # Train the model with your dataset
# results = model.train(
#     data='data.yaml',
#     epochs=10,  # Reduced epochs for faster training
#     imgsz=320,  # Reduced image size to save VRAM
#     batch=2,    # Very small batch size to fit in 2GB VRAM
# )

# Load your trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Export to ONNX
model.export(format="onnx")