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
from dnn_utils.embedding import FeatureExtractor

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
    for det in dets:
        tlbr = det.to_tlbr().astype(np.int32) - 1
        tlbr = np.clip(tlbr, 0, clip_max)
        crop = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]]
        feature = feature_extractor(crop)
        det.set_feature(feature)

    return dets


def run(sequence_dir, detector_path, quantized, use_pytorch,
        reid_path, output_file, min_confidence, nms_max_overlap,
        max_feature_distance, nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir)

    object_detector = ObjectDetector(
        model_path=detector_path, quantized=quantized, use_pytorch=use_pytorch,
        img_size=(seq_info['image_size'][1], seq_info['image_size'][0])
    )

    feature_extractor = FeatureExtractor(
        model_path=reid_path, use_pytorch=use_pytorch
    )

    metric = nn_matching.NearestNeighborDistanceMetric(
        "euclidean", max_feature_distance, nn_budget)

    tracker = Tracker(metric)
    results = []

    def frame_callback(vis, frame_idx):
        # print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            object_detector, feature_extractor, seq_info,
            frame_idx, min_confidence, nms_max_overlap)

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
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
    if display:
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
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detector_path", help="Path to custom object detector.", default=None)
    parser.add_argument(
        "--quantized", help="The object detector is quantized.",
        default=False, type=bool_string)
    parser.add_argument(
        "--use_pytorch", help="Use pytorch to run the model.",
        default=False, type=bool_string)
    parser.add_argument(
        "--reid_path", help="Path to custom ReID network.", default=None)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.5, type=float)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=0.5, type=float)
    parser.add_argument(
        "--max_feature_distance", help="Gating threshold for feature distance "
        "metric (object appearance).", type=float, default=200)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.sequence_dir, args.detector_path, args.quantized,
        args.use_pytorch, args.reid_path, args.output_file,
        args.min_confidence, args.nms_max_overlap,
        args.max_feature_distance, args.nn_budget, args.display)
