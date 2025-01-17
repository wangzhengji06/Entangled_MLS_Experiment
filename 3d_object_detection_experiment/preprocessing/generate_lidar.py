import argparse
import os

import numpy as np
import scipy.misc as ssc

import kitti_util


def project_disp_to_points(calib, disp, max_high, max_depth):
    disp[disp < 0] = 0
    baseline = 0.54
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1. - mask)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 0] < max_depth) & (cloud[:, 2] < max_high)
    return cloud[valid]

def project_depth_to_points(calib, depth, max_high, max_depth):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 0] < max_depth) & (cloud[:, 2] < max_high)
    return cloud[valid]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Libar') 
    # it seems that you will have to use calib, which is a ground truth
    parser.add_argument('--calib_dir', type=str,
                        default='~/Kitti/object/training/calib')
    # of couse you will have to use your disparity file
    parser.add_argument('--disparity_dir', type=str,
                        default='~/Kitti/object/training/predicted_disparity')
    # saved location
    parser.add_argument('--save_dir', type=str,
                        default='~/Kitti/object/training/predicted_velodyne')
    # some parameters 
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--max_depth', type=int, default=200)
    parser.add_argument('--is_depth', action='store_true')

    args = parser.parse_args()

    assert os.path.isdir(args.disparity_dir)
    assert os.path.isdir(args.calib_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    # you just get a bunch of file names such as 000020.npy
    disps = [x for x in os.listdir(args.disparity_dir) if x[-3:] == 'png' or x[-3:] == 'npy']
    disps = sorted(disps)


    for fn in disps:
        # get the actual digit 
        predix = fn[:-4]

        # get the calib file
        calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
        calib = kitti_util.Calibration(calib_file)
        # disp_map = ssc.imread(args.disparity_dir + '/' + fn) / 256.
        if fn[-3:] == 'png':
            disp_map = ssc.imread(args.disparity_dir + '/' + fn)
        #get the disparity file
        elif fn[-3:] == 'npy':
            disp_map = np.load(args.disparity_dir + '/' + fn)
        else:
            assert False
        if not args.is_depth:
            disp_map = (disp_map*256).astype(np.uint16)/256.
            lidar = project_disp_to_points(calib, disp_map, args.max_high, args.max_depth)
        else:
            disp_map = (disp_map).astype(np.float32)/256.
            lidar = project_depth_to_points(calib, disp_map, args.max_high, args.max_depth)
        # pad 1 in the indensity dimension
        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
        lidar = lidar.astype(np.float32)
        lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))
        print('Finish Depth {}'.format(predix))
