import csv
 
train_temp_path = './Dataset/temp.csv'
train = []
val = []
max_num = 5

cnt = 0
with open(train_temp_path) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    for row in csv_reader:       
        if int(row[7]) != -1 and int(row[8]) <= max_num:
            cnt += 1
            if cnt % 5 == 0:
                val.append(row)
            else:
                train.append(row)
     

with open('./Dataset/train_temp.csv',"w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerows(train)
   
with open('./Dataset/val_temp.csv',"w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerows(val)
