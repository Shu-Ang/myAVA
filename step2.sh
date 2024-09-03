split=$1
gpu=$2
set -e

cleanup() {
    echo "error! start cleanning up"
    bash clean2.sh
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

if [ ! $gpu ];then
    gpu=0
fi

bash utils/distribute_via.sh $split

if [ ! -f "./Dataset/data_without_personID.csv" ]; then
    echo "start transforming via_json to csv......"
    python ./utils/json_extract.py
fi

if [ ! -f "./Dataset/data_with_personID.csv" ]; then
    echo "start detecting with deepsort......"
    python ./yolovDeepsort/yolov5_to_deepsort.py --gpu ${gpu}
fi

if [ ! -f "./Dataset/temp.csv" ]; then
    echo "start concatenating data_with_personID.csv and data_without_personID.csv......"
    python ./utils/concat.py
fi

if [ -f "./Dataset/temp.csv" ]; then
    if [ $split = 'train' ];then
        python ./utils/generate_final_proposals.py --train True
    else
        python ./utils/generate_final_proposals.py --val True
    fi
fi

if [ ! -d "./Dataset/rawframes" ]; then
    mkdir ./Dataset/rawframes
fi

if [ ! -z "$(ls ./Dataset/frames/)" ]; then
    mv ./Dataset/frames/* ./Dataset/rawframes/
fi

python ./utils/change_raw_frames.py