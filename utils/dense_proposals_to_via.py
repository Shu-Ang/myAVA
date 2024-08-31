from via3_tool import Via3Json
import pickle
from collections import defaultdict
import os
import cv2
import re
import json


avaMin_dense_proposals_path = "./Dataset/dense_proposals.pkl"

root_path = "./Dataset/choose_frames/"

action_list_path = "./Dataset/action_list.json"

with open(avaMin_dense_proposals_path,'rb') as f:
    info = pickle.load(f, encoding='iso-8859-1') 

with open(action_list_path, 'r') as file:
    attributes_dict = json.load(file)
    
videos = [entry for entry in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, entry))]
for video in videos:
    frame_path = os.path.join(root_path, video)
    json_path = frame_path + "/" + video + "_proposal.json"
    via3 = Via3Json(json_path, mode='dump')
    images = os.listdir(frame_path)
    
    # 使用正则表达式提取下划线后的数字，并按照这个数字进行排序
    sorted_images = sorted(images, key=lambda x: int(re.search(r'_(\d+).jpg$', x).group(1)))
    vid_list = list(map(str,range(1, len(sorted_images)+1)))
    via3.dumpPrejects(vid_list)
    via3.dumpConfigs()
    via3.dumpAttributes(attributes_dict)
    files_dict,  metadata_dicts = {},{}
    image_id = 1
    for image in sorted_images:
        img = cv2.imread(os.path.join(frame_path, image))  #读取图片信息
        sp = img.shape #[高|宽|像素值由三种原色构成]
        img_H = sp[0]
        img_W = sp[1]
        vid , _ = os.path.splitext(image)
        files_dict[str(image_id)] = dict(fname=image, type=2)
        
        result = info[vid]
        box_id = 1
        for data in result:
            xyxy = data
            xyxy[0] = img_W*xyxy[0]
            xyxy[2] = img_W*xyxy[2]
            xyxy[1] = img_H*xyxy[1]
            xyxy[3] = img_H*xyxy[3]
            temp_w = xyxy[2] - xyxy[0]
            temp_h = xyxy[3] - xyxy[1]
            metadata_dict = dict(vid=str(image_id),
                                xy=[2, float(xyxy[0]), float(xyxy[1]), float(temp_w), float(temp_h), float(xyxy[4])],
                                av={'1': '0'})
            metadata_dicts['{}_{}'.format(vid,box_id)] = metadata_dict
            box_id = box_id + 1
        image_id = image_id + 1
    via3.dumpFiles(files_dict)
    via3.dumpMetedatas(metadata_dicts)
    views_dict = {}
    for i, vid in enumerate(vid_list,1):
        views_dict[vid] = defaultdict(list)
        views_dict[vid]['fid_list'].append(str(i))
    via3.dumpViews(views_dict)
    via3.dempJsonSave()



