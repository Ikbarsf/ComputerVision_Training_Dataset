from ultralytics import YOLO
import cv2

imgPath = "./ikan-goby.jpg"
modelPath = './runs/classify/train/weights/best.pt'

image = cv2.imread(imgPath)
model = YOLO(modelPath)
results = model(image)

for r in results:
    print(r.show())
