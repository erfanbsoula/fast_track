import numpy as np
import cv2
import time
from dnn_utils.object_detection import ObjectDetector
from dnn_utils.embedding import FeatureExtractor
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric

FRAME_RATE = 5
FRAME_DURATION = 1 / FRAME_RATE
REID_BUDGET = 5

object_detector = ObjectDetector(
    model_path='dnn_utils/models/yolov8n_quant.onnx',
    img_size=(640, 480)
)

feature_extractor = FeatureExtractor(
    model_path='dnn_utils/models/osnet_x0_25.onnx',
)

def draw_prediction(img, class_id, tlbr):

    label = str(class_id)
    color = [0, 0, 255]
    cv2.rectangle(img, tlbr[:2], tlbr[2:], color, 2)
    cv2.putText(img, label, (tlbr[0]+5, tlbr[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

fps = []
yolo = []
osnet = []
track_time = []

def display_stats():
    print('fps:', sum(fps)/len(fps))
    print('avg yolo time:', sum(yolo)/len(yolo))
    print('avg reid time:', sum(osnet)/len(osnet))
    print('avg tracker time:', sum(track_time)/len(track_time))


metric = NearestNeighborDistanceMetric(
    metric='euclidean', matching_threshold=210, budget=30)

tracker = Tracker(metric)

prev_time_point = 0
proc_duration = 0
capture = cv2.VideoCapture(0)

while True:

    t_loop_start = time.time()
    time_elapsed = t_loop_start - prev_time_point
    isTrue, frame = capture.read()

    if time_elapsed >= FRAME_DURATION - proc_duration:

        tmp = time.time()
        dets = object_detector(frame)
        yolo.append(time.time()-tmp)

        tmp = time.time()
        for det in dets:
            x1, y1, x2, y2 = np.maximum(0, det.to_tlbr().astype(np.int32))
            det.set_feature(feature_extractor(frame[y1:y2, x1:x2]))
        osnet.append(time.time()-tmp)

        tmp = time.time()
        tracker.update(dets)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            draw_prediction(frame, track.track_id, track.to_tlbr().astype(np.int32))

        tracker.predict()
        track_time.append(time.time()-tmp)

        cv2.imshow("object detection", frame)
        t_now = time.time()
        fps.append(1/(t_now-prev_time_point))
        prev_time_point = t_now
        proc_duration = t_now - t_loop_start + 0.001

    if cv2.waitKey(1) & 0xFF == ord('x'):
        display_stats()
        break

capture.release()
cv2.destroyAllWindows()