import numpy as np
import cv2
import time
from dnn_utils.object_detection import ObjectDetector
from dnn_utils.reidentification import FeatureExtractor
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric


FRAME_RATE = 8
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

def draw_prediction(img, class_id, tlbr):

    label = str(class_id)
    color = [0, 0, 255]
    if class_id < 5:
        color = [0, 255, 0]
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
    metric='euclidean',
    matching_threshold=120,
    budget=40
)

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

        clip_max = np.array([frame.shape[1], frame.shape[0]]*2, dtype=np.int32)
        clip_max = clip_max - 1

        frames = []
        for det in dets:
            tlbr = det.to_tlbr().astype(np.int32)
            tlbr = np.clip(tlbr, 0, clip_max)
            x1, y1, x2, y2 = tlbr
            frames.append(frame[y1:y2, x1:x2])

        if len(frames) != 0:
            features = feature_extractor(frames)
            for i, det in enumerate(dets):
                det.set_feature(features[i])

        osnet.append(time.time()-tmp)

        tmp = time.time()
        tracker.update(dets)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            draw_prediction(frame, track.known_id, track.to_tlbr().astype(np.int32))


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