import json
import csv
import os

train_action_cnt_path = './Dataset/train_action_cnt.json'
val_action_cnt_path = './Dataset/val_action_cnt.json'
train_data_path = './Dataset/annotations/train.csv'
val_data_path = './Dataset/annotations/val.csv'

if os.path.exists(train_data_path):
    cnt = [0] * 7
    with open(train_action_cnt_path, 'r') as jsonfile:
        json_data = json.load(jsonfile)
        
    with open(train_data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            actionID = row[6]
            cnt[int(actionID) - 1] += 1

    for i in range(0, 7):
        json_data[str(i + 1)]['cnt'] = cnt[i]
        
    with open(train_action_cnt_path, 'w') as jsonfile:
        json.dump(json_data, jsonfile, ensure_ascii=False, indent=4)
    
if os.path.exists(val_data_path):
    cnt = [0] * 7
    with open(val_action_cnt_path, 'r') as jsonfile:
        json_data = json.load(jsonfile)
        
    with open(val_data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            actionID = row[6]
            cnt[int(actionID) - 1] += 1

    for i in range(0, 7):
        json_data[str(i + 1)]['cnt'] = cnt[i]
        
    with open(val_action_cnt_path, 'w') as jsonfile:
        json.dump(json_data, jsonfile, ensure_ascii=False, indent=4)