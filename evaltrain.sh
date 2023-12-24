!/bin/bash
for i in {02,04,05,09,10,11,13}
do
    echo "running tracker on train $i ..."
    python deep_sort_app.py --sequence_dir=./MOT16/train/MOT16-$i \
        --output_file=./MOT16/results/MOT16-$i.txt \
        --min_confidence=0.3 --nn_budget=30 --display=False # &> /dev/null

done

echo "copying the results to TrackEval directory ..."
for i in {02,04,05,09,10,11,13}
do
    cp ./MOT16/results/MOT16-$i.txt ./TrackEval/data/trackers/mot_challenge/MOT16-train/fast_track/data/
done

python ./TrackEval/scripts/run_mot_challenge.py \
    --BENCHMARK MOT16 \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL fast_track \
    --GT_FOLDER ./evaluation/ \
    --TRACKERS_FOLDER ./evaluation/ \
    --METRICS CLEAR \
    --USE_PARALLEL False --NUM_PARALLEL_CORES 1
