#!/bin/bash

# first check if directories exist
if ! [ -d ./evaluation/tracking_result ]; then
  mkdir ./evaluation/tracking_result
fi

if ! [ -d ./evaluation/tracking_result/MOT16-train ]; then
  mkdir ./evaluation/tracking_result/MOT16-train
fi

if ! [ -d ./evaluation/tracking_result/MOT16-train/fast_track ]; then
  mkdir ./evaluation/tracking_result/MOT16-train/fast_track
fi

if ! [ -d ./evaluation/tracking_result/MOT16-train/fast_track/data ]; then
  mkdir ./evaluation/tracking_result/MOT16-train/fast_track/data
fi


for i in {02,04,05,09,10,11,13}
do
    echo "running tracker on train $i ..."
    python deep_sort_app.py \
        --sequence_dir ./MOT16/train/MOT16-$i \
        --detector_path ./dnn_utils/models/$1 \
        --quantized $2 \
        --reid_path ./dnn_utils/models/$3 \
        --output_file ./evaluation/tracking_result/MOT16-train/fast_track/data/MOT16-$i.txt \
        --min_confidence 0.4 \
        --nms_max_overlap 0.5 \
        --nn_budget 100 \
        --display False # &> /dev/null
done

python ./TrackEval/scripts/run_mot_challenge.py \
    --BENCHMARK MOT16 \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL fast_track \
    --GT_FOLDER ./evaluation/ground_truth/ \
    --TRACKERS_FOLDER ./evaluation/tracking_result/ \
    --METRICS CLEAR \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 1
