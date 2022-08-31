import argparse
import glob
import os.path
import re
import matplotlib.pyplot as plt
import cv2
import numpy as np

from exr_analyse import load_rectify_depth, mono_disp_to_depth
from funcs import disp_to_depth


def load_raft_depth(args, test):
    mask = cv2.imread(r"p35_l.png")[:, :, 0]
    depth_list = []
    depth_gt_list = []

    if not test:
        dire = sorted(glob.glob(os.path.join(args.test_dir, r"*.npy")))
    else:
        out = compare()
        dire = [os.path.join(args.test_dir, name) for name in out]

    for npy in dire:
        disp_raw = np.load(npy)
        disp_raw = disp_raw * -1
        disp_raw[disp_raw <= 0] = 0.1
        depth = disp_to_depth(disp_raw, 61.0, 530.277)
        depth[mask == 0] = 0.1

        # plt.imshow(depth, cmap='jet', vmax=8000)
        # plt.colorbar()
        # plt.show()
        depth_list.append(depth)

        index = re.split(r"[_.]", npy)[-2]
        rectified_depth_gt = load_rectify_depth("l", index)
        rectified_depth_gt[mask == 0] = 0.1

        # plt.imshow(rectified_depth_gt, cmap='jet', vmax=8000)
        # plt.colorbar()
        # plt.show()

        depth_gt_list.append(rectified_depth_gt)

    depth = np.asarray(depth_list)
    depth_gt = np.asarray(depth_gt_list)
    np.savez(f"{args.test_dir.split('/')[-1]}_rendered.npz", *depth)
    np.savez(f"{args.test_dir.split('/')[-1]}_rendered_gt.npz", *depth_gt)


def load_mono_depth(args):
    mask = cv2.imread(r"p35_l.png")[:, :, 0]
    depth_list = []
    depth_gt_list = []
    for npy in sorted(glob.glob(os.path.join(args.test_dir, r"*.npy"))):
        disp_raw = np.load(npy)
        disp_raw = np.reshape(disp_raw, (480, 640))
        disp_raw[disp_raw <= 0] = 0.1
        depth = 1 / disp_raw
        depth[mask == 0] = 0.1

        # plt.imshow(depth, cmap='jet', vmax=8000)
        # plt.colorbar()
        # plt.show()
        depth_list.append(depth)

        index = re.split(r"[_]", npy)[-3]
        rectified_depth_gt = load_rectify_depth("l", index)
        rectified_depth_gt[mask == 0] = 0.1

        # plt.imshow(rectified_depth_gt, cmap='jet', vmax=8000)
        # plt.colorbar()
        # plt.show()

        depth_gt_list.append(rectified_depth_gt)

    depth = np.asarray(depth_list)
    depth_gt = np.asarray(depth_gt_list)
    np.savez(f"{args.test_dir.split('/')[-1]}_rendered.npz", *depth)
    np.savez(f"{args.test_dir.split('/')[-1]}_rendered_gt.npz", *depth_gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate npz files for evaluation in MonoDepth2 model.")
    parser.add_argument("--test_dir",
                        default=r"/Users/fyli/Documents/Msc_Computing/Individual_project/monodepth2/test_results_1119")
    args = parser.parse_args()
    # load_raft_depth(args, False)
    load_mono_depth(args)
