import pickle
import numpy as np
import csv
import os

dense_proposals_train_path = "./Dataset/annotations/dense_proposals_train.pkl"
dense_proposals_val_path = "./Dataset/annotations/dense_proposals_val.pkl"
train_data_path = './Dataset/train_temp.csv'
val_data_path = './Dataset/val_temp.csv'
dense_proposals_train = {}
dense_proposals_val = {}
data_train = {}
data_val = {}
train = []
val = []

with open(train_data_path, "r") as train_data:
    train_reader = csv.reader(train_data)
    for row in train_reader:
        # 处理每一行数据
        key = f"{row[0]},{int(row[1]):04d}"
        list = [row[2], row[3], row[4], row[5], row[6]]
        train.append([row[0], row[1], row[2], row[3], row[4], row[5], row[7], row[8]])
        if key not in data_train:
            data_train[key] = []
        data_train[key].append(list)
        
for key, value_list in data_train.items():
    dense_proposals_train[key] = np.array(value_list, dtype=np.float64)
    

with open('./Dataset/annotations/dense_proposals_train.pkl',"ab") as pklfile: 
    pickle.dump(dense_proposals_train, pklfile)

with open('./Dataset/annotations/train.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(train)
        
        
with open(val_data_path, "r") as val_data:
    val_reader = csv.reader(val_data)
    for row in val_reader:
        # 处理每一行数据
        key = f"{row[0]},{int(row[1]):04d}"
        list = [row[2], row[3], row[4], row[5], row[6]]
        val.append([row[0], row[1], row[2], row[3], row[4], row[5], row[7], row[8]])
        if key not in data_val:
            data_val[key] = []
        data_val[key].append(list)
    
for key, value_list in data_val.items():
    dense_proposals_val[key] = np.array(value_list, dtype=np.float64)
    
# 保存为pkl文件
with open('./Dataset/annotations/dense_proposals_val.pkl',"ab") as pklfile: 
    pickle.dump(dense_proposals_val, pklfile)

with open('./Dataset/annotations/val.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(val)
