echo "start cutting frames......"
bash ./Dataset/cut_frames.sh


echo "start choosing frames......"
bash ./Dataset/choose_frames.sh


if [ -z "$(ls ./yolovDeepsort/yolov5/runs/detect)" ]; then
    echo "start detecting with yolov5......"
    python ./yolovDeepsort/yolov5/detect.py --source ./Dataset/choose_frames_all/ --save-txt --save-conf 
fi

if [ ! -f "./yolovDeepsort/mywork/dense_proposals_train.pkl" ]; then
    echo "save detect results as ./yolovDeepsort/mywork/dense_proposals_train.pkl"
    python ./yolovDeepsort/mywork/dense_proposals_train.py
fi

echo "start transforming pkl to via......"
python ./yolovDeepsort/mywork/dense_proposals_train_to_via.py
echo "start processing via......"
python ./Dataset/change_via_json.py 

if [ ! -f "./Dataset/choose_frames.zip" ]; then
    echo "start zipping choose_frames......"
    zip -r ./Dataset/choose_frames.zip ./Dataset/choose_frames
fi

echo "you can start annotating now!"