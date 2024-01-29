import os
import numpy as np
import cv2
import time
from dnn_utils.object_detection import ObjectDetector
from dnn_utils.reidentification import FeatureExtractor
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric


FRAME_RATE = 5
FRAME_DURATION = 1 / FRAME_RATE

object_detector = ObjectDetector(
    engine='yolo', device='cpu',
    model_path='dnn_utils/models/yolov8n_crowdhuman.pt',
    model_image_size=(320, 512),
    target_cls=1
)

feature_extractor = FeatureExtractor(
    engine='torchreid', device='cpu',
    model_name='osnet_x0_25',
    model_path='dnn_utils/models/osnet_x0_25_market.pth'
)

database = 'database/ids.npy'

if os.path.isfile(database):
    features = np.load(database)
else:
    features = np.empty(shape=(0, 512), dtype=np.float32)

def get_area(det):

    tlbr = det.to_tlbr().astype(np.int32)
    x1, y1, x2, y2 = tlbr
    return abs(x2 - x1) * abs(y2 - y1)


def draw_prediction(img, class_id, tlbr):

    label = str(class_id)
    color = [0, 0, 255]
    cv2.rectangle(img, tlbr[:2], tlbr[2:], color, 2)
    cv2.putText(img, label, (tlbr[0]+5, tlbr[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


prev_time_point = 0
proc_duration = 0
capture = cv2.VideoCapture(0)

while True:

    t_loop_start = time.time()
    time_elapsed = t_loop_start - prev_time_point
    isTrue, frame = capture.read()

    if time_elapsed >= FRAME_DURATION - proc_duration:

        dets = object_detector(frame, nms_th=0.7)
        dets = sorted(dets, key=get_area, reverse=True)

        clip_max = np.array([frame.shape[1], frame.shape[0]]*2, dtype=np.int32)
        clip_max = clip_max - 1

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord('t'):

            tlbr = dets[0].to_tlbr().astype(np.int32)
            tlbr = np.clip(tlbr, 0, clip_max)
            x1, y1, x2, y2 = tlbr
            imgs = [frame[y1:y2, x1:x2]]
            cv2.imshow("image taken", imgs[0])
            feature_new = feature_extractor(imgs)
            features = np.concatenate((features, feature_new), axis=0)
            features = features[-5:]
            np.save(database, features, allow_pickle=False)

        elif keypress == ord('x'):
            break

        draw_prediction(frame, 0, dets[0].to_tlbr().astype(np.int32))
        cv2.imshow("object detection", frame)

        t_now = time.time()
        prev_time_point = t_now
        proc_duration = t_now - t_loop_start + 0.001


capture.release()
cv2.destroyAllWindows()