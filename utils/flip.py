import cv2
import csv
import os
import pickle
import numpy as np

rawframes_path = 'Dataset/rawframes'
train_data_path = '/Dataset/annotations/train.csv'
val_data_path = 'Dataset/annotations/val.csv'
train_flip_data_path = '/Dataset/annotations/train_flip.csv'
val_flip_data_path = 'Dataset/annotations/val_flip.csv'
dense_proposals_train_path = 'Dataset/annotations/dense_proposals_train.pkl'
dense_proposals_val_path = 'Dataset/annotations/dense_proposals_val.pkl'
dense_proposals_train_flip_path = 'Dataset/annotations/dense_proposals_train_flip.pkl'
dense_proposals_val_flip_path = 'Dataset/annotations/dense_proposals_val_flip.pkl'

print('start flipping frames')
videos = os.listdir(rawframes_path)
for video in videos:
    print("start flipping " + video)
    video_path = os.path.join(rawframes_path, video)
    if 'flip' not in video:
        if video.endswith('val'):
            flip_video_path = video_path.rsplit('_', 1)[0] + '_flip_val'
        else:
            flip_video_path = video_path + '_flip'
        
        if not os.path.isdir(flip_video_path):
            os.mkdir(flip_video_path)
            for root, dirs, files in os.walk(video_path):
                for img in files:
                    img_path = os.path.join(video_path, img)
                    image = cv2.imread(img_path)
                    horizontally_flipped_image = cv2.flip(image, 1)
                    cv2.imwrite(os.path.join(flip_video_path, img), horizontally_flipped_image)
                    
print("done!")

print('start flipping annotations')
flip_train_data = []
with open(train_data_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader: 
        video_name, sec, x1, y1, x2, y2, actionID, personID = row
        if 'flip' not in video_name:
            flip_video_name = video_name + '_flip'
            flip_x1 = float(1.0 - float(x2))
            flip_x2 = float(1.0 - float(x1))
            flip_train_data.append(row)
            flip_train_data.append([flip_video_name, sec, flip_x1 , y1, flip_x2, y2, actionID, personID])

with open(train_flip_data_path, "w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerows(flip_train_data)

flip_val_data = []
with open(val_data_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader: 
        video_name, sec, x1, y1, x2, y2, actionID, personID = row
        if 'flip' not in video_name:
            flip_video_name = video_name.rsplit('_', 1)[0] + '_flip_val'
            flip_x1 = float(1.0 - float(x2))
            flip_x2 = float(1.0 - float(x1))
            flip_val_data.append(row)
            flip_val_data.append([flip_video_name, sec, flip_x1 , y1, flip_x2, y2, actionID, personID])

with open(val_flip_data_path, "w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerows(flip_val_data)
    
print("done!")
print('start flipping proposals')    
with open(dense_proposals_train_path,'rb') as f:
    train_proposals = pickle.load(f, encoding='iso-8859-1') 

train_proposals_flip = {}
for key, value in train_proposals.items():
    video_name = key.split(',')[0]
    sec = key.split(',')[1]
    key_flip = video_name + '_flip,' + sec
    value_flip = []
    for proposal in value:
       x1, y1, x2, y2, conf = proposal[0], proposal[1], proposal[2], proposal[3], proposal[4]
       value_flip.append([1.0 - x2, y1, 1.0 - x1, y2, conf])
    train_proposals_flip[key_flip] = np.array(value_flip, dtype=np.float64)

train_proposals.update(train_proposals_flip)
# 保存为pkl文件
with open(dense_proposals_train_flip_path,"wb") as pklfile: 
    pickle.dump(train_proposals, pklfile)



with open(dense_proposals_val_path,'rb') as f:
    val_proposals = pickle.load(f, encoding='iso-8859-1') 

val_proposals_flip = {}
for key, value in val_proposals.items():
    video_name = key.split(',')[0]
    sec = key.split(',')[1]
    key_flip = video_name.rsplit('_', 1)[0] + '_flip_val,' + sec
    value_flip = []
    for proposal in value:
       x1, y1, x2, y2, conf = proposal[0], proposal[1], proposal[2], proposal[3], proposal[4]
       value_flip.append([1.0 - x2, y1, 1.0 - x1, y2, conf])
    val_proposals_flip[key_flip] = np.array(value_flip, dtype=np.float64)

val_proposals.update(val_proposals_flip)

# 保存为pkl文件
with open(dense_proposals_val_flip_path, "wb") as pklfile: 
    pickle.dump(val_proposals, pklfile)

print("done!")