from ultralytics import YOLO

# Carregar o modelo YOLO
model = YOLO("yolov8s_playing_cards.pt")

class_names = model.names
print("Classes detectadas:")
for idx, name in enumerate(class_names):
    print(f"{idx}: {name}")

model.predict(source = 0, show=True)



