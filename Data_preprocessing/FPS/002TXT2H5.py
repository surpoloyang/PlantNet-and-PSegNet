import os
import numpy as np
import h5py
import argparse

def loadDataFile(path):
    data = np.loadtxt(path)
    point_xyz = data[:, 0:3]
    point_rgb = data[:, 3:6].astype(int)
    label = (data[:, 6::]).astype(int)
    return point_xyz, point_rgb, label


def change_scale(data):

    xyz_min = np.min(data[:, 0:3], axis=0)
    xyz_max = np.max(data[:, 0:3], axis=0)
    xyz_move = xyz_min + (xyz_max - xyz_min) / 2
    data[:, 0:3] = data[:, 0:3] - xyz_move
    # scale
    scale = np.max(data[:, 0:3])
    return data[:, 0:3] / scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    FLAGS = parser.parse_args()
    DATA_ALL = []
    num_sample = 4096
    base_path = f'./PlantNet/PlantNet_pytorch/data/FPSprocessed/{FLAGS.mode}_aug'
    DATA_FILES = os.listdir(base_path)
    h5_root = f'./PlantNet/PlantNet_pytorch/data/FPSprocessed/{FLAGS.mode}_h5'
    if not os.path.exists(h5_root):
        os.makedirs(h5_root)
    for fn in range(len(DATA_FILES)):
         current_data, current_feature, current_label= loadDataFile(os.path.join(base_path, DATA_FILES[fn]))
         change_data = change_scale(current_data)
         data_feature_label = np.column_stack((change_data, current_feature, current_label))
         DATA_ALL.append(data_feature_label)

    output = np.vstack(DATA_ALL)
    output = output.reshape(-1, num_sample, 8)
    # f = h5py.File('./PepperSeedlings/train_h5/train.h5', "w")
    f = h5py.File(os.path.join(h5_root, f'{FLAGS.mode}.h5'), "w")
    f['data'] = output[:, :, 0:3]
    f['rgb'] = output[:, :, 3:6]
    f['pid'] = output[:, :, 6]  # 实例标签
    f['seglabel'] = output[:, :, 7]  # 语义标签
    # f['obj'] = output[:, :, 5]

    f.close()



