from ultralytics import YOLO
import cv2

# Carregar o modelo YOLO
model = YOLO("yolov8s_playing_cards.pt")

model.predict(source = 0, show=True)