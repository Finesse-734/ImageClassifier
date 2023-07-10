from ultralytics import YOLO
import numpy as np

model = YOLO('/Users/nikunj/runs/classify/train8/weights/last.pt')  # load a custom model

results = model('/Users/nikunj/Downloads/rain_img.jpeg')  # predict on an image

names_dict = results[0].names

probs = results[0].probs.tolist()

# print(names_dict)
# print(probs)
print(names_dict[np.argmax(probs)])

