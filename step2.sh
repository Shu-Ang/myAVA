bash distribute.sh

if [ ! -f "./Dataset/train_without_personID.csv" ]; then
    echo "start transforming via_json to csv......"
    python ./utils/json_extract.py
fi

if [ ! -f "./Dataset/train_with_personID.csv" ]; then
    echo "start detecting with deepsort......"
    python ./yolovDeepsort/yolov5_to_deepsort.py
fi

if [ ! -f "./Dataset/temp.csv" ]; then
    echo "start concatenating train_with_personID.csv and train_without_personID.csv......"
    python ./utils/concat.py
fi

if [ -f "./Dataset/temp.csv" ]; then
    python ./utils/generate_dense_proposals.py --train True
fi

if [ ! -d "./Dataset/rawframes" ]; then
    mkdir ./Dataset/rawframes
fi

if [ ! -z "$(ls ./Dataset/frames/)" ]; then
    mv ./Dataset/frames/* ./Dataset/rawframes/
fi

python ./utils/change_raw_frames.py
pyton ./utils/cnt_actions.py

