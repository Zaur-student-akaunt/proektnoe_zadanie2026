from ultralytics import YOLO

model = YOLO("runs/detect/garbage_detector/weights/best.pt")

results = model.predict(
    source="test.jpg",
    conf=0.4,
    save=True
)

print("Готово")
