import argparse
import sys
import csv
import os
import re
#sys.path.append('..') #目的是为了导入上一级的yolov5
sys.path.insert(0, './yolovDeepsort/yolov5/')

from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size,xyxy2xywh
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import pickle
from PIL import Image

#python yolov5_to_deepsort.py 

# dict存放最后的json
datas = []
def detect(opt):
    root_path = "./Dataset/choose_frames"
    
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    
    # 这里是dense_proposals_train_deepsort.pkl的路径，
    with open('./Dataset/dense_proposals_train_deepsort.pkl','rb') as f:
        info = pickle.load(f, encoding='iso-8859-1') 
    
    videos = [entry for entry in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, entry))]
    for video in videos:
        video_path = os.path.join(root_path, video)
        
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
        
        files = os.listdir(video_path)
        images = [entry for entry in files if entry.lower().endswith('.jpg')]
        sorted_images = sorted(images, key=lambda x: int(re.search(r'_(\d+).jpg$', x).group(1)))
        # 读取当前标注信息所对应的图片
        for image in sorted_images:    
            image_path = os.path.join(video_path, image)
            tempImg = cv2.imread(image_path)
            imageId , _ = os.path.splitext(image)
            sec = int((int(imageId.split('_')[1]) - 1) / 30)
            dets = info[imageId]
            if dets != []:
                # 获取图片的大小
                imgsz = tempImg.shape
                # pkl中的的坐标是左上角与右下角，即xyxy，
                # 但是输入到deepsort中的值是人的中心坐标与长宽，注意是中心坐标，即xywh
                xyxys = torch.FloatTensor(len(dets), 4)
                confs = torch.FloatTensor(len(dets))
                clss = torch.FloatTensor(len(dets))
                for index, det in enumerate(dets):
                    xyxys[index][0]=det[0]*imgsz[1]
                    xyxys[index][1]=det[1]*imgsz[0]
                    xyxys[index][2]=det[2]*imgsz[1]
                    xyxys[index][3]=det[3]*imgsz[0]
                    confs[index]=(float(det[4]))
                    clss[index]=0.
                    
                xywhs = xyxy2xywh(xyxys)
                
                img = np.array(Image.open(image_path))
                outputs = deepsort.update(xywhs, confs, clss, img)
                
                if len(outputs) > 0:
                    for output in outputs:
                        x1 = output[0]/ imgsz[1]
                        y1 = output[1]/ imgsz[0]
                        x2 = output[2]/ imgsz[1]
                        y2 = output[3]/ imgsz[0]
                        data = [video,sec,x1,y1,x2,y2,output[4]]
                        datas.append(data)
                        
    with open('./Dataset/train_with_personID.csv',"w") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows(datas)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_deepsort", type=str, default="./yolovDeepsort/deep_sort_pytorch/configs/deep_sort.yaml")
    opt = parser.parse_args()
    with torch.no_grad():
        detect(opt)