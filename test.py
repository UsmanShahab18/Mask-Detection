from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train2/weights/best.pt")
print("Classes:", model.names)

img = cv2.imread("1.jpg")  # put one of your test images here
results = model.predict(source=img, conf=0.05, imgsz=320)

results[0].show() 