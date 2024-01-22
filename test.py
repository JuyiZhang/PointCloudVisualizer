from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-seg.pt")

image = cv2.imread("data/Session_1705938368/10.5.11.225/1705938/1705938383854_ab_image.png")
result = model.track(image)