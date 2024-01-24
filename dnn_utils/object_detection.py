from dnn_utils.transforms import LetterBox
from deep_sort.detection import Detection
import numpy as np
import cv2

class ObjectDetector:

    def __init__(self, engine, model_path,
                 input_image_size=(640, 960), # (Height, Width)
                 model_image_size=(640, 640), # (Height, Width)
                 device='cpu', quantized=False,
                 human_cls_id=0):

        assert isinstance(engine, str)

        self.model_path = model_path
        self.input_image_size = input_image_size
        self.model_image_size = model_image_size
        self.device = device
        self.quantized = quantized
        self.human_cls_id = human_cls_id

        if engine == 'yolo':
            self.init_yolo()

        elif engine == 'deepsparse':
            self.init_deepsparse()

        else: raise Exception("unknown engine!")

        self.engine_type = engine


    def __call__(self, img, conf_th=0.5, nms_th=0.5):

        return self.process(img, conf_th, nms_th)
    

    def init_yolo(self):

        from ultralytics import YOLO

        self.engine = YOLO(self.model_path)
        self.engine.fuse()

        self.process = self.run_yolo


    def run_yolo(self, img, conf_th, nms_th):

        res = self.engine(
            img, imgsz = self.model_image_size,
            conf=conf_th, iou=nms_th, verbose=False
        )

        res = res[0]

        indices = res.boxes.cls == self.human_cls_id
        boxes = res.boxes.xywh[indices].cpu().numpy()
        confs = res.boxes.conf[indices].cpu().numpy()

        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes = boxes.astype(np.int32)

        detections = [
            Detection(boxes[i], confs[i]) for i in range(len(boxes))]

        return detections


    def init_deepsparse(self):

        from deepsparse import Engine

        self.engine = Engine(
            model = self.model_path,
            num_cores=1
        )

        self.dtype = np.uint8 if self.quantized else np.float32
        self.transform = LetterBox(
            shape = self.input_image_size[::-1],
            new_shape = self.model_image_size
        )

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
            if confidence > conf_th and class_id == self.human_cls_id:
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
