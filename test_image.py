import numpy as np
import time
import cv2
from PIL import Image
from dnn_utils.object_detection import ObjectDetector

image = "basilica.jpg"

img = Image.open(image)
img = np.array(img, dtype=np.uint8)

object_detector = ObjectDetector(
    model_path='dnn_utils/models/yolov8-m-coco-base.onnx',
    quantized=False, use_pytorch=True,
    img_size=(img.shape[1], img.shape[0])
)

def draw_prediction(img, tlbr):
    color = [0, 0, 255]
    cv2.rectangle(img, tlbr[:2], tlbr[2:], color, 2)

t1 = time.time()
for i in range(10):
    dets = object_detector(img)
print(time.time()-t1)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for det in dets:
    draw_prediction(img, det.to_tlbr().astype(np.int32))

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
