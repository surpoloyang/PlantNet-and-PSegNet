import numpy as np
import math
import time
import sys
import os
import argparse

class FarthestSampler:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)  # Returns the sum of squared Euclidean distances between the set of sample points and other points

    def _call__(self, pts, k):  # PTS is the input point cloud,  K is the number of downsampling
        farthest_pts = np.zeros((k, 8),
                                dtype=np.float32)  # The first three columns are coordinates xyz, then RGB, and the seventh column is instance label，the eighth column is the semantic label.
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self._calc_distances(farthest_pts[0, :3], pts[:, :3])
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self._calc_distances(farthest_pts[i, :3], pts[:, :3]))
        return farthest_pts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    FLAGS = parser.parse_args()
    path = f'./PlantNet/PlantNet_pytorch/data/FPSprocessed/{FLAGS.mode}'
    saved_path = f'./PlantNet/PlantNet_pytorch/data/FPSprocessed/{FLAGS.mode}_aug'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    Filelist = os.listdir(path)
    n = len(Filelist)
    for idx in range(n):
        points = np.loadtxt(os.path.join(path, Filelist[idx]), dtype=float, delimiter=' ')
        pcd_array = np.array(points)
        print("pcd_array.shape:", pcd_array.shape)
        sample_count = 4096
        for z in range(10):  # do 10 times data augmentation
            # Fixed number of points after FPS downsampling
            # Perform FPS Downsampling for center point set and edge point set respectively
            FPS = FarthestSampler()
            sample_points = FPS._call__(pcd_array, sample_count)
            file_nameR = Filelist[idx].split(".")[0] + "_aug_" + str(z) + ".txt"
            np.savetxt(os.path.join(saved_path, file_nameR), sample_points, fmt='%f %f %f %d %d %d %d %d')
