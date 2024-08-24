import json
import csv

action_cnt_path = './Dataset/action_cnt.json'
train_data_path = './Dataset/annotations/train.csv'
cnt = 0 * range(0, 8)
with open(action_cnt_path, 'r', encoding='utf-8') as jsonfile:
    json_data = json.load(jsonfile)
    
with open(train_data_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        actionID = row[6]
        cnt[actionID - 1] += 1

for i in range(0, 8):
    json_data[str(i + 1)]['cnt'] = cnt[i]
    
with open(action_cnt_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(json_data, jsonfile, indent=4)