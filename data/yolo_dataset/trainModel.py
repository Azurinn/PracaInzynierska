from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    name='stain_detection',
    patience=15,
    save=True,
    plots=True
)