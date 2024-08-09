import sys
import os
import json
import pickle

#传参 labelPath是yolov5检测结果的位置，需要获取0（0代表人）的四个坐标值，还需要检测概率
# ../yolov5/runs/detect/exp/labels
labelPath = "./yolovDeepsort/yolov5/runs/detect/exp/labels"

#传参 保存为pkl的地址，这是像ava数据集对齐
# ./avaMin_dense_proposals_train.pkl
avaMin_dense_proposals_path = "./yolovDeepsort/mywork/dense_proposals_train.pkl"

results_dict = {}
for root, dirs, files in os.walk(labelPath):
    if root == labelPath:
        for file in files:    
            #读取yolov5中的信息
            key = file.split('.')[0] + "." + file.split('.')[1]
            with open(os.path.join(root, file)) as temp_txt:
                temp_data_txt = temp_txt.readlines() 
                results = []
                for i in temp_data_txt:
                    # 只要人的信息
                    j = i.split(' ')
                    if j[0]=='0':
                
                        # 由于yolov5的检测结果是 xywh
                        # 要将xywh转化成xyxy
                        y = j
                        y[1] = float(j[1]) - float(j[3]) / 2  # top left x
                        y[2] = float(j[2]) - float(j[4]) / 2  # top left y
                        y[3] = float(j[1]) + float(j[3])  # bottom right x
                        y[4] = float(j[2]) + float(j[4])  # bottom right y
                        
                        results.append([y[1],y[2],y[3],y[4],y[5]])
            results_dict[key] = results

# 保存为pkl文件
with open(avaMin_dense_proposals_path,"wb") as pklfile: 
    pickle.dump(results_dict, pklfile)
    