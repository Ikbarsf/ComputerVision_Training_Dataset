from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

results = model.train(data="C:/UNEJ/Perkuliahan/Semester 6/Computer Vision/Dataset_Ikan/FishImgDataset", epochs=50, imgsz=128)