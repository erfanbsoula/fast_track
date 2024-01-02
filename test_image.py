import numpy as np
import time
import cv2
from PIL import Image
from dnn_utils.object_detection import ObjectDetector

image = "basilica.jpg"

img = Image.open(image)
img = np.array(img, dtype=np.uint8)

object_detector = ObjectDetector(
    engine='deepsparse',
    model_path='dnn_utils/models/yolov8-n-coco-base.onnx',
    img_size=(640, 960)
)

def draw_prediction(img, tlbr):
    color = [0, 0, 255]
    cv2.rectangle(img, tlbr[:2], tlbr[2:], color, 2)

n_iterations = 10
t = time.time()

for i in range(n_iterations):
    dets = object_detector(img, conf_th=0.25, nms_th=0.75)

t = (time.time() - t) / n_iterations
print(f'average inference time: {t:.3f}')

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for det in dets:
    draw_prediction(img, det.to_tlbr().astype(np.int32))

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
