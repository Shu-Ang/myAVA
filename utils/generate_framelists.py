import csv
import os
import shutil

frames_dir = './Dataset/rawframes'
out_dir = './Dataset/frame_lists'
train_annotation = './Dataset/annotations/train.csv'
val_annotation = './Dataset/annotations/val.csv'
train_dict = {}
val_dict = {}
train_list = []
val_list = []
unuse = []
title = ['original_vido_id', 'video_id', 'frame_id', 'path', 'labels']
train_list.append(title)
val_list.append(title)
id = 0

if os.path.exists(train_annotation):
    with open(train_annotation, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            key = row[0]
            if key not in train_dict:
                train_dict[key] = []

if os.path.exists(val_annotation):
    with open(val_annotation, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            key = row[0]
            if key not in val_dict:
                val_dict[key] = []
                  
for root, dirs, files in os.walk(frames_dir):
    for dir in dirs:
        video_path = os.path.join(frames_dir, dir)
        if dir not in val_dict and dir not in train_dict:
            shutil.rmtree(video_path)
        else:
            images = os.listdir(video_path)
            if dir.endswith('val'):
                
                for image in images:
                    frame_id = int(image.split('.')[0].split('_')[1])
                    frame_path = os.path.join(dir, image)
                    val_list.append([dir, id, frame_id, frame_path, '\'\''])
            else:
                for image in images:
                    frame_id = int(image.split('.')[0].split('_')[1])
                    frame_path = os.path.join(dir, image)     
                    train_list.append([dir, id, frame_id, frame_path, '\'\''])  
            id += 1
        
with open(os.path.join(out_dir, 'train.csv'), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerows(train_list)
    
with open(os.path.join(out_dir, 'val.csv'), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerows(val_list)
    