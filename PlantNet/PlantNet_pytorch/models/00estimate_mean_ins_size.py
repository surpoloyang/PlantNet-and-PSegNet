# To estimate the mean instance size of each class in training set
import os
import sys
import numpy as np
from scipy import stats
import argparse
import provider

def estimate(args):
    MODEL_DIR = os.path.dirname(os.path.abspath(__file__))   # PlantNet/PlantNet_pytorch/models
    sys.path.append(MODEL_DIR)
    BASE_DIR = os.path.dirname(MODEL_DIR)                    # PlantNet/PlantNet_pytorch
    LOG_DIR = os.path.join(BASE_DIR, args.log) # PlantNet/PlantNet_pytorch/log
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    num_classes = 40
    file_path = "data/train_file_list.txt"

    train_file_list = provider.getDataFiles(os.path.join(BASE_DIR,file_path))

    mean_ins_size = np.zeros(num_classes)
    ptsnum_in_gt = [[] for itmp in range(num_classes)]

    for h5_filename in train_file_list:
        cur_data, cur_feature, cur_group, _, cur_sem = provider.loadDataFile_with_groupseglabel_stanfordindoor(os.path.abspath(h5_filename))
        for i in range(cur_data.shape[0]):
            cur_group_batch = np.reshape(cur_group[i,...], [-1]) #327680x1
            cur_sem_batch = np.reshape(cur_sem[i,...], [-1])
    
            un = np.unique(cur_group_batch)
            for ig, g in enumerate(un):
                tmp = (cur_group_batch == g)
                sem_seg_g = int(stats.mode(cur_sem_batch[tmp])[0])
                ptsnum_in_gt[sem_seg_g].append(np.sum(tmp))

    for idx in range(num_classes):
        if ptsnum_in_gt[idx] != []:
            mean_ins_size[idx] = np.mean(ptsnum_in_gt[idx]).astype(int)

    print(mean_ins_size)
    np.savetxt(os.path.join(LOG_DIR, 'mean_ins_size.txt'),mean_ins_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='log', help='Log dir [default: ]')
    FLAGS = parser.parse_args()
    estimate(FLAGS)
