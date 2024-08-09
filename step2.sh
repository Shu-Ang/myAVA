
echo "start transforming annotations to csv......"
python ./Dataset/json_extract.py



echo "start detecting with deepsort......"
python ./yolovDeepsort/yolov5_to_deepsort.py



echo "start concatenating train_with_personID.csv and train_without_personID.csv......"
python ./Dataset/concat.py



echo "adding data to train.csv......"
python ./Dataset/process_train_temp.py

mv ./Dataset/frames/* ./Dataset/rawframes/



