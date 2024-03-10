# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
from time import time

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from dnn_utils.object_detection import ObjectDetector
from dnn_utils.reidentification import FeatureExtractor

def gather_sequence_info(sequence_dir):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}

    groundtruth = None
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    detections = None
    detection_file = os.path.join(sequence_dir, "det/det.txt")
    if os.path.exists(detection_file):
        detections = np.loadtxt(detection_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(object_detector, feature_extractor,
                      seq_info, frame_idx, conf_th, nms_th):
    """Create detections for given frame index.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """

    image = cv2.imread(
        seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

    clip_max = np.array(seq_info['image_size'][::-1]*2, dtype=np.int32)
    clip_max = clip_max - 1

    dets = object_detector(image, conf_th, nms_th)

    frames = []
    for det in dets:
        tlbr = det.to_tlbr().astype(np.int32)
        tlbr = np.clip(tlbr, 0, clip_max)
        x1, y1, x2, y2 = tlbr
        frames.append(image[y1:y2, x1:x2])

    features = feature_extractor(frames)
    for i, det in enumerate(dets):
        det.set_feature(features[i])

    return dets


def run(args):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    args : dict
        Configuration dictionary.
    """
    seq_info = gather_sequence_info(args['sequence_dir'])

    image_size = [
        round(seq_info['image_size'][0] / 32) * 32,
        round(seq_info['image_size'][1] / 32) * 32,
    ]

    print(seq_info["image_size"])

    object_detector = ObjectDetector(
        engine=args['detector_engine'],
        model_path=args['detector_path'],
        input_image_size=seq_info['image_size'],
        model_image_size=image_size,
        quantized=args['quantized'],
        device=args['device'],
        target_cls=args['target_cls']
    )

    feature_extractor = FeatureExtractor(
        engine=args['reid_engine'],
        model_name=args['reid_model'],
        model_path=args['reid_path'],
        device=args['device']
    )

    metric = nn_matching.NearestNeighborDistanceMetric(
        "euclidean",
        args['max_feature_distance'],
        args['max_gallery_size']
    )

    tracker = Tracker(
        metric, max_iou_distance=args['max_IoU_distance']
    )
    results = []

    def frame_callback(vis, frame_idx):
        # print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            object_detector, feature_extractor, seq_info,
            frame_idx, args['min_confidence'], args['nms_max_overlap']
        )

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if args['display']:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if args['display']:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)

    start = time()
    visualizer.run(frame_callback)
    finish = time()

    total_frames = seq_info["max_frame_idx"] - seq_info["min_frame_idx"] + 1
    fps = total_frames / (finish - start)
    print(f"fps: {fps:.1f}")

    # Store results.
    f = open(args['output_file'], 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")

    parser.add_argument(
        "--sequence_dir",
        help="Path to MOTChallenge sequence directory",
        default=None, required=True
    )
    parser.add_argument(
        "--device",
        help="The device to run Neural Network inference (cuda | cpu)",
        default='cuda'
    )
    parser.add_argument(
        "--detector_path",
        help="Path to custom object detector",
        default='./dnn_utils/models/yolov8n_crowdhuman.pt'
    )
    parser.add_argument(
        "--detector_engine",
        help="The engine for running object detection",
        default='yolo'
    )
    parser.add_argument(
        "--quantized",
        help="Shows the object detector is quantized",
        action='store_true'
    )
    parser.add_argument(
        "--target_cls",
        help="Target object class for tracking.",
        type=int, default=1
    )
    parser.add_argument(
        "--reid_model",
        help="Model name of the custom ReID network",
        default='osnet_x0_25'
    )
    parser.add_argument(
        "--reid_path",
        help="Path to custom ReID network",
        default='./dnn_utils/models/osnet_x0_25_market.pth'
    )
    parser.add_argument(
        "--reid_engine",
        help="The engine for running ReID network",
        default='torchreid'
    )
    parser.add_argument(
        "--output_file",
        help="Path to the tracking output file. This file will " \
             "contain the tracking results on completion.",
        default="./tmp/hypothesis.txt"
    )
    parser.add_argument(
        "--min_confidence",
        help="Detection confidence threshold. Discard detections " \
             "that have a confidence lower than this value.",
        type=float, default=0.5
    )
    parser.add_argument(
        "--nms_max_overlap", 
        help="Non-maximum suppression threshold. Discard detections " \
             "that overlap with an IoU more than this value.",
        type=float, default=0.7
    )
    parser.add_argument(
        "--max_feature_distance",
        help="Gating threshold for appearance feature distance. " \
             "Longer distances will not be considered in appearance matching.",
        type=float, default=120.
    )
    parser.add_argument(
        "--max_IoU_distance",
        help="Maximum IoU distance accepted for IoU matching.",
        type=float, default=0.7
    )
    parser.add_argument(
        "--max_gallery_size",
        help="Maximum size of the appearance descriptor gallery."
             "If None, no budget is enforced.",
        type=int, default=30
    )
    parser.add_argument(
        "--display",
        help="Show intermediate tracking results",
        action='store_true'
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    run(args)
