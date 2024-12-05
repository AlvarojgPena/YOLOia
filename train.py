from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # pass any model type

model.train(data = "dataset_custom.yaml", imgsz = 640, batch = 8, workers = 0, device = 0, epochs= 5)

