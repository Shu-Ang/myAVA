import json
import os
import csv
import cv2
import pickle


dense_proposals_path = "./Dataset/dense_proposals_train_deepsort.pkl"
results_dict = {}
# dict存放最后的json
dicts = []
# 通过循环与判断来找出via的json标注文件
videos = [entry for entry in os.listdir("./Dataset/choose_frames") if not entry.startswith('.')]
for video in videos:
    video_path = os.path.join("./Dataset/choose_frames", video)
    for root, dirs, files in os.walk(video_path):
        for file in files:
            #via的json标注文件以_proposal.json结尾
            if "_finish.json" in file:
                jsonPath = root+'/'+file
                #读取标注文件
                with open(jsonPath, encoding='utf-8') as f:
                    line = f.readline()
                    viaJson = json.loads(line)
                    
                    attributes = viaJson['attribute']
                    attributeNums = [0]
                    for i in range(1, len(attributes) + 1):
                        attributeNums.append(len(attributes[str(i)]['options']) + attributeNums[i - 1])
                    
                    files = {}
                    for file in viaJson['file']:
                        fid = viaJson['file'][file]['fid']
                        fname = viaJson['file'][file]['fname']
                        files[fid]=fname
                        prefix_name , _ = os.path.splitext(fname)
                        results_dict[prefix_name] = []
                        
                    for metadata in viaJson['metadata']:
                        imagen_x = viaJson['metadata'][metadata]
                        #获取人的坐标
                        xy = imagen_x['xy'][1:]
                        # 忽略小框，防止手误
                        if xy[2] < 10 or xy[3] < 10:
                            continue
                        #获取vid，目的是让坐标信息与图片名称、视频名称对应
                        vid = imagen_x['vid']
                        fname = files[vid]
                        #获取视频帧ID
                        imageId, _ = os.path.splitext(fname)
                        videoId = imageId.rsplit('_', 1)[0]
                        sec = int((int(imageId.rsplit('_', 1)[1]) - 1) / 30)
                        # 获取坐标对应的图片，因为最后的坐标值需要在0到1
                        # 就需要用现有坐标值/图片大小
                        imgPath = root + '/' + fname
                        imgTemp = cv2.imread(imgPath)  #读取图片信息
                        sp = imgTemp.shape #[高|宽|像素值由三种原色构成]
                        img_H = sp[0]
                        img_W = sp[1]
                        x1 = xy[0] / img_W
                        y1 = xy[1] / img_H
                        x2 = (xy[0]+xy[2]) / img_W
                        y2 = (xy[1]+xy[3]) / img_H
                        
                        # 防止坐标点超过图片大小
                        if x1 < 0:
                            x1 = 0
                        if x1 > 1:
                            x1 = 1
                            
                        if x2 < 0:
                            x2 = 0
                        if x2 > 1:
                            x2 = 1
                            
                        if y1 < 0:
                            y1 = 0
                        if y1 > 1:
                            y1 = 1
                            
                        if y2 < 0:
                            y2 = 0
                        if y2 > 1:
                            y2 = 1
                            
                        confidence = xy[4] if len(xy) == 5 else float(0.9)
                        results_dict[imageId].append([x1,y1,x2,y2,confidence])
                            
                        for action in imagen_x['av']:
                            avs = imagen_x['av'][action]
                            #行为复选框不为空,获取复选框中的行为
                            if avs != '':
                                #一个复选框可能有多个选择
                                avArr = avs.split(',')
                                for av in avArr:                                                  
                                    actionId = attributeNums[int(action)-1]+int(av)+1
                                    dict = [videoId,sec,x1,y1,x2,y2,confidence,actionId]
                                    dicts.append(dict)
                            
with open('./Dataset/train_without_personID.csv',"w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerows(dicts)   
    
# 保存为pkl文件
with open(dense_proposals_path,"wb") as pklfile: 
    pickle.dump(results_dict, pklfile)