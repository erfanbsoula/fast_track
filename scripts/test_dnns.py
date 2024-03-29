# run this script from the base directory
import sys
sys.path.append(".")

import numpy as np
import time
import cv2
from PIL import Image
from dnn_utils.object_detection import ObjectDetector
from dnn_utils.reidentification import FeatureExtractor


image = "./scripts/basilica.jpg"

img = Image.open(image)
img = np.array(img, dtype=np.uint8)

object_detector = ObjectDetector(
    engine='yolo', device='cpu',
    model_path='dnn_utils/models/yolov8n.pt',
    input_image_size=img.shape[:2],
    model_image_size=(320, 512),
    target_cls=0
)

feature_extractor = FeatureExtractor(
    engine='torchreid', device='cpu',
    model_name='osnet_x0_25',
    model_path='dnn_utils/models/osnet_x0_25_market.pth',
)

def draw_prediction(img, tlbr):
    color = [0, 0, 255]
    cv2.rectangle(img, tlbr[:2], tlbr[2:], color, 2)

n_iterations = 1
t = time.time()

for i in range(n_iterations):
    dets = object_detector(img, conf_th=0.4, nms_th=0.7)

t = (time.time() - t) / n_iterations
print(f'average object detection time: {t*1000:.1f} ms')
print(f'detecttion count: {len(dets)}')

frames = []

for det in dets:
    x1, y1, x2, y2 = np.maximum(0, det.to_tlbr().astype(np.int32))
    frames.append(img[y1:y2, x1:x2])

n_iterations = 1
t = time.time()

for i in range(n_iterations):
    features = feature_extractor(frames)

t = (time.time() - t) / len(frames) / n_iterations
print(f'average extraction time per img: {t*1000:.1f} ms')

print(np.array(features).shape)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for det in dets:
    draw_prediction(img, det.to_tlbr().astype(np.int32))

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
