split=$1

set -e

cleanup() {
    echo "error! start cleanning up"
    bash clean1.sh
    # 执行清理操作，例如关闭文件描述符、删除临时文件等
}

trap cleanup ERR

if [ ! $split = 'train' ] && [ ! $split = 'val' ];then
    echo arg must be 'train' or 'val'
    exit 1
fi

if [ ! $split ];then
    echo arg can not be void
    exit 1
fi

if [ $split = 'val' ];then
    bash ./utils/change_name.sh
fi

echo "start cutting frames......"
bash ./utils/cut_frames.sh Dataset/${split}_videos


echo "start choosing frames......"
bash ./utils/choose_frames.sh Dataset/${split}_videos


if [ -z "$(ls ./yolovDeepsort/yolov5/runs/detect/exp/labels)" ]; then
    echo "start detecting with yolov5......"
    python ./yolovDeepsort/yolov5/detect.py --source ./Dataset/choose_frames_all/ --save-txt --save-conf 
fi

if [ ! -f "./Dataset/dense_proposals.pkl" ]; then
    echo "save detect results as ./Dataset/dense_proposals.pkl"
    python ./utils/generate_dense_proposals.py
fi

echo "start transforming pkl to via......"
python ./utils/dense_proposals_to_via.py

echo "start processing via......"
python ./utils/change_via_json.py 

if [ -f "./Dataset/choose_frames.zip" ]; then
    rm -rf ./Dataset/choose_frames.zip
fi

echo "start zipping choose_frames......"
zip -r ./Dataset/choose_frames.zip ./Dataset/choose_frames

echo "you can start annotating now!"