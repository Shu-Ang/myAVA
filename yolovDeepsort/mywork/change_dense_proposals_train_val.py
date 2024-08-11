import pickle
import numpy as np
import csv

with open('./yolovDeepsort/mywork/dense_proposals_train_deepsort.pkl','rb') as f:
    info = pickle.load(f, encoding='iso-8859-1') 
    
train_data_path = './Dataset/annotations/train.csv'
val_data_path = './Dataset/annotations/val.csv'
dense_proposals_train = {}
dense_proposals_val = {}

with open(train_data_path, "r") as train_data:
    train_reader = csv.reader(train_data)
    for row in train_reader:
        # 处理每一行数据
        key = f"{row[0]}_{(int(row[1]) - 1) * 30:06d}"
        if key in dense_proposals_train:
            list = [row[2], row[3], row[4], row[5]]
            dense_proposals_train[key].append(list)
        else:
            dense_proposals_train[key] = []
            dense_proposals_train[key].append(list)
    
    # 保存为pkl文件
    with open('./Dataset/annotations/dense_proposals_train.pkl',"wb") as pklfile: 
        pickle.dump(dense_proposals_train, pklfile)
        
        
with open(val_data_path, "r") as val_data:
    val_reader = csv.reader(val_data)
    for row in val_reader:
        # 处理每一行数据
        key = row[0]
        if key in dense_proposals_val:
            list = [row[2], row[3], row[4], row[5]]
            dense_proposals_val[key].append(list)
        else:
            dense_proposals_val[key] = []
            dense_proposals_val[key].append(list)
    
    # 保存为pkl文件
    with open('./Dataset/annotations/dense_proposals_val.pkl',"wb") as pklfile: 
        pickle.dump(dense_proposals_val, pklfile)


