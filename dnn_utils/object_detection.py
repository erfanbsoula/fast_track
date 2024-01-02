from dnn_utils.transforms import LetterBox
from deep_sort.detection import Detection
import numpy as np
import cv2

class ObjectDetector:

    def __init__(self, engine, model_path, img_size,
                 model_input_size=(640, 640), quantized=False):

        assert isinstance(engine, str)

        if engine == 'yolo':
            self.init_yolo(model_path, img_size)

        elif engine == 'deepsparse':
            self.init_deepsparse(
                model_path, quantized, img_size, model_input_size)

        else: raise Exception("unknown engine!")


    def __call__(self, img, conf_th=0.5, nms_th=0.5):

        return self.process(img, conf_th, nms_th)
    

    def init_yolo(self, model_path, img_size):
        
        from ultralytics import YOLO

        self.engine = YOLO(model_path)

        self.imgsz = img_size

        self.process = self.run_yolo


    def run_yolo(self, img, conf_th, nms_th):

        res = self.engine(
            img, imgsz=self.imgsz, conf=conf_th, iou=nms_th,
            verbose=False
        )

        res = res[0]

        indices = res.boxes.cls == 0
        boxes = res.boxes.xywh[indices].cpu().numpy()
        confs = res.boxes.conf[indices].cpu().numpy()

        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes = boxes.astype(np.int32)

        detections = [
            Detection(boxes[i], confs[i]) for i in range(len(boxes))]

        return detections


    def init_deepsparse(self, model_path, quantized, img_size,
                        model_input_size):

        from deepsparse import Engine

        self.engine = Engine(model=model_path, num_cores=1)

        self.quantized = quantized
        self.dtype = np.uint8 if quantized else np.float32
        self.transform = LetterBox(
            shape=img_size[::-1], new_shape=model_input_size)

        self.process = self.run_deepsparse


    def run_deepsparse(self, img, conf_th, nms_th):

        img = self.transform(img)
        img = img.transpose((2, 0, 1)) # whc -> cwh
        img = np.expand_dims(img, axis=0) # add batch dim

        if not self.quantized:
            img = img  / 255.0 # normalize the input

        img = np.ascontiguousarray(img, dtype=self.dtype)

        output = self.engine([img])[0].squeeze(axis=0).transpose((1, 0))

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

        for detection in output:
            process_detection(detection)
        
        boxes = np.array(boxes, dtype=np.int32)
        boxes = self.transform.unbox(boxes)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)
        detections = [
            Detection(boxes[i], confidences[i]) for i in indices]

        return detections
