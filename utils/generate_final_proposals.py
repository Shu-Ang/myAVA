import pickle
import numpy as np
import csv
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('--train', type = bool, help='train')
    parser.add_argument('--val', type = bool, help='val')
    args = parser.parse_args()
    return args

def main():    
    args = parse_args()
    if args.train:
        split = 'train'
    else:
        split = 'val'
        
    dense_proposals_path = "./Dataset/annotations/dense_proposals_" + split + '.pkl'
    data_path = './Dataset/temp.csv'
    out_path = './Dataset/annotations/' + split + '.csv' 
    dense_proposals = {}
    data_dict = {}
    data = []

    with open(data_path, "r") as train_data:
        train_reader = csv.reader(train_data)
        for row in train_reader:
            # 处理每一行数据
            key = f"{row[0]},{int(row[1]):04d}"
            list = [row[2], row[3], row[4], row[5], row[6]]
            data.append([row[0], row[1], row[2], row[3], row[4], row[5], row[7], row[8]])
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(list)
            
    for key, value_list in data_dict.items():
        dense_proposals[key] = np.array(value_list, dtype=np.float64)
        
    if os.path.exists(dense_proposals_path):
        with open(dense_proposals_path, 'rb') as f:
            info = pickle.load(f, encoding='iso-8859-1') 
            dense_proposals.update(info)
            
    with open(dense_proposals_path,"wb") as pklfile: 
        pickle.dump(dense_proposals, pklfile)

    with open(out_path, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
        
if __name__ == '__main__':
    main()        