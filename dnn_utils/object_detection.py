from deepsparse import Engine
from dnn_utils.transforms import LetterBox
from deep_sort.detection import Detection
import numpy as np
import cv2

class ObjectDetector:

    def __init__(self, model_path, img_size, model_input_size=(640, 640)):

        self.engine = Engine(model=model_path)
        self.transform = LetterBox(
            shape=img_size, new_shape=model_input_size)

    def __call__(self, img, conf_th=0.3, nms_th=0.3):

        img = self.transform(img)
        img = img.transpose((2,0,1)) # whc -> cwh
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.

        outputs = self.engine([img])[0].transpose((0, 2, 1))

        boxes = []
        confidences = []

        def process_detection(detection):
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_th and class_id == 0:
                x = int(detection[0] - detection[2]/2)
                y = int(detection[1] - detection[3]/2)
                w = int(detection[2])
                h = int(detection[3])
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

        for out in outputs:
            for detection in out:
                process_detection(detection)
        
        boxes = np.array(boxes, dtype=np.int32)
        boxes = self.transform.unbox(boxes)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)
        detections = [
            Detection(boxes[i], confidences[i]) for i in indices]

        return detections
