from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

model.train(data='/Users/nikunj/Downloads/dataset2', epochs=20, imgsz=64)
