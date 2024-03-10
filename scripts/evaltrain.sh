#!/bin/bash

# first check if directory exists
if ! [ -d ./evaluation/tracker_results/MOT16/MOT16-train/fast_track/data ]; then
  mkdir -p ./evaluation/tracker_results/MOT16/MOT16-train/fast_track/data
fi

for i in {02,04,05,09,,10,11,13}
do
    echo "running tracker on train $i ..."
    python deep_sort_app.py \
        --sequence_dir ./evaluation/datasets/MOT16/train/MOT16-$i \
        --output_file ./evaluation/tracking_result/MOT16-train/fast_track/data/MOT16-$i.txt \
        --device cpu # &> /dev/null
done

python ./evaluation/TrackEval/scripts/run_mot_challenge.py \
    --BENCHMARK MOT16 \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL fast_track \
    --GT_FOLDER ./evaluation/ground_truth/ \
    --TRACKERS_FOLDER ./evaluation/tracking_result/ \
    --METRICS CLEAR \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 1
