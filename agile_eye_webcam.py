import numpy as np
import cv2
import time
from dnn_utils.object_detection import ObjectDetector
from dnn_utils.reidentification import FeatureExtractor
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from load_database import get_known_ids
import json

import dynamixel_sdk as dxl
from agile_eye.Motor import Dynamixel_MX_106, Dynamixel_MX_64, MotorGroup
from agile_eye.AgileEye import ikp

DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 57600

HOME_POSITIONS = np.array([2583, 2065, 2064])

curr_x_angle = 0
curr_y_angle = 0
start_position = HOME_POSITIONS + ikp(0, curr_x_angle, curr_y_angle)

portHandler = dxl.PortHandler(DEVICENAME)

if portHandler.openPort():
    print("succeeded to open the port")
else:
    print("failed to open the port")
    quit()

if portHandler.setBaudRate(BAUDRATE):
    print("succeeded to change the baudrate")
else:
    print("failed to change the baudrate")
    quit()

motors = MotorGroup([
    Dynamixel_MX_64(portHandler, 1, start_position[0]),
    Dynamixel_MX_106(portHandler, 2, start_position[1]),
    Dynamixel_MX_106(portHandler, 3, start_position[2]),
])

FRAME_RATE = 5
FRAME_DURATION = 1 / FRAME_RATE

object_detector = ObjectDetector(
    engine='yolo', device='cpu',
    model_path='dnn_utils/models/yolov8n.pt',
    model_image_size=(320, 512),
    target_cls=0
)

feature_extractor = FeatureExtractor(
    engine='torchreid', device='cpu',
    model_name='osnet_x0_25',
    model_path='dnn_utils/models/osnet_x0_25_market.pth'
)

def get_area(det):

    tlbr = det.to_tlbr().astype(np.int32)
    x1, y1, x2, y2 = tlbr
    return abs(x2 - x1) * abs(y2 - y1)

def draw_prediction(img, track):
    
    tlbr = track.to_tlbr().astype(np.int32)
    label = str(track.track_id)
    color = [0, 0, 255]
    if track.known_id > 0:
        color = [0, 255, 0]
        label = track.name

    cv2.rectangle(img, tlbr[:2], tlbr[2:], color, 2)
    cv2.putText(img, label, (tlbr[0]+5, tlbr[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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

known_ids, known_features, known_names = get_known_ids(feature_extractor)

tracker = Tracker(metric, known_ids, known_features, known_names)
tracker.kf._std_weight_position = 1 / 20
tracker.kf._std_weight_velocity = 1 / 30

def controller(target_track):

    global curr_x_angle, curr_y_angle

    target = target_track.to_tlwh()
    x, y = target[:2] + target[2:] / 2

    step_size = 2
    changed = False

    if x < frame.shape[1] / 2 - 60:
        
        gain = int(abs(x - frame.shape[1] / 2) // 100)
        if curr_x_angle - step_size >= -20:
            curr_x_angle -= step_size * gain
            changed = True
        
    elif x > frame.shape[1] / 2 + 60:
        
        gain = int(abs(x - frame.shape[1] / 2) // 100)
        if curr_x_angle + step_size <= 20:
            curr_x_angle += step_size * gain
            changed = True

    if target[1] < 25:
        if curr_y_angle - step_size >= -30:
            curr_y_angle -= step_size
            changed = True

    elif target[1] > frame.shape[0] / 2 - 140:
        if curr_y_angle + step_size <= 20:
            curr_y_angle += step_size
            changed = True

    if changed:
        motors.setGoalPositions(list(HOME_POSITIONS + ikp(0, curr_x_angle, curr_y_angle)))


prev_time_point = 0
proc_duration = 0
capture = cv2.VideoCapture(2)

frame_counter = 0

while True:

    t_loop_start = time.time()
    time_elapsed = t_loop_start - prev_time_point
    isTrue, frame = capture.read()

    if time_elapsed >= FRAME_DURATION - proc_duration:

        frame_counter += 1

        tmp = time.time()
        dets = object_detector(frame)
        dets = sorted(dets, key=get_area, reverse=True)
        dets = dets[:5]
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
            draw_prediction(frame, track)

        tracker.predict()

        target_track = None
        target_priority = 1e6

        for track in tracker.tracks:
            if not track.is_confirmed():
                continue

            if(track.known_id > 0 and
               track.time_since_update < 3 and
               track.known_id < target_priority):
                
                target_track = track
                target_priority = track.known_id

        if target_track != None:
            controller(target_track)
            frame_counter = 0
        
        if frame_counter > 2 * FRAME_RATE:
            motors.setGoalPositions(list(start_position))
            frame_counter = 0

        track_time.append(time.time()-tmp)

        cv2.imshow("object detection", frame)
        t_now = time.time()
        fps.append(1/(t_now-prev_time_point))
        prev_time_point = t_now
        proc_duration = t_now - t_loop_start + 0.001

    if cv2.waitKey(1) & 0xFF == ord('x'):
        display_stats()
        history = [tracker.history[key] for key in tracker.history]
        history = 'data = ' + json.dumps(history)
        with open('interface/data.js', 'w') as f:
            f.write(history) 
        break

capture.release()
cv2.destroyAllWindows()
