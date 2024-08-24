echo "start cutting frames......"
bash ./utils/cut_frames.sh Dataset/test_videos


echo "start choosing frames......"
bash ./utils/choose_frames.sh Dataset/test_videos


if [ -z "$(ls ./yolovDeepsort/yolov5/runs/detect/exp/labels)" ]; then
    echo "start detecting with yolov5......"
    python ./yolovDeepsort/yolov5/detect.py --source ./Dataset/choose_frames_all/ --save-txt --save-conf 
fi

if [ ! -f "./Dataset/dense_proposals_train.pkl" ]; then
    echo "save detect results as ./Dataset/dense_proposals_train.pkl"
    python ./utils/dense_proposals_train.py
fi

echo "start transforming pkl to via......"
python ./utils/dense_proposals_train_to_via.py

echo "start processing via......"
python ./utils/change_via_json.py 

if [ ! -f "./Dataset/choose_frames.zip" ]; then
    echo "start zipping choose_frames......"
    zip -r ./Dataset/choose_frames.zip ./Dataset/choose_frames
fi

echo "you can start annotating now!"