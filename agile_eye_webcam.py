import numpy as np
import cv2
import time
from dnn_utils.object_detection import ObjectDetector
from dnn_utils.reidentification import FeatureExtractor
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric

import dynamixel_sdk as dxl
from agile_eye.Motor import Dynamixel_MX_106, Dynamixel_MX_64, MotorGroup
from agile_eye.AgileEye import ikp

DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 57600

HOME_POSITIONS = np.array([2583, 2065, 2064])

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
    Dynamixel_MX_64(portHandler, 1, HOME_POSITIONS[0]),
    Dynamixel_MX_106(portHandler, 2, HOME_POSITIONS[1]),
    Dynamixel_MX_106(portHandler, 3, HOME_POSITIONS[2]),
])

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
capture = cv2.VideoCapture(2)

curr_x_angle = 0
curr_y_angle = 0
frame_counter = 0

while True:

    t_loop_start = time.time()
    time_elapsed = t_loop_start - prev_time_point
    isTrue, frame = capture.read()

    if time_elapsed >= FRAME_DURATION - proc_duration:
        
        frame_counter += 1

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
        
        features = feature_extractor(frames)
        for i, det in enumerate(dets):
            det.set_feature(features[i])

        osnet.append(time.time()-tmp)

        tmp = time.time()
        tracker.update(dets)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            draw_prediction(frame, track.track_id, track.to_tlbr().astype(np.int32))
        
        # if frame_counter % (FRAME_RATE // 3) == 0:
        if True:

            try:
                target = tracker.tracks[0].to_tlwh()
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
                
                elif target[1] > frame.shape[0] / 2 - 100:
                    if curr_y_angle + step_size <= 20:
                        curr_y_angle += step_size
                        changed = True

                if changed:
                    motors.setGoalPositions(list(HOME_POSITIONS + ikp(0, curr_x_angle, curr_y_angle)))

            except: pass

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