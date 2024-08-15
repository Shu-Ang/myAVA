import csv
import json

train_with_personID_path = './Dataset/train_with_personID.csv'
train_without_personID_path = './Dataset/train_without_personID.csv'
anciton_cnt_path = './Dataset/action_cnt.json'
train_with_personID = []
train_without_personID = []

with open(train_with_personID_path) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    for row in csv_reader:           
        train_with_personID.append(row)

with open(train_without_personID_path) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    for row in csv_reader:          
        train_without_personID.append(row)

with open(anciton_cnt_path, 'r') as jsonfile:
    cnt = json.load(jsonfile)
    
dicts = []
for data in train_with_personID:
    for temp_data in train_without_personID:
        # 属于同一个视频同一张图片
        if data[0] == temp_data[0] and int(data[1]) == int(temp_data[1]):
            if abs(float(data[2])-float(temp_data[2]))<0.05 and abs(float(data[3])-float(temp_data[3]))<0.05 and abs(float(data[4])-float(temp_data[4]))<0.05 and abs(float(data[5])-float(temp_data[5]))<0.05:
                # data[6]-1 代表将ID-1，原因是ID从0开始计数
                # temp_data[6] confidence
                # temp_data[7] actionID
                cnt[str(temp_data[7])]["cnt"] += 1
                dict = [data[0],data[1],data[2],data[3],data[4],data[5],temp_data[6],temp_data[7],int(data[6])-1]
                dicts.append(dict)

with open('./Dataset/temp.csv',"w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerows(dicts)

with open(anciton_cnt_path, "w") as jsonfile:
    json.dump(cnt, jsonfile, ensure_ascii=False, indent=4)