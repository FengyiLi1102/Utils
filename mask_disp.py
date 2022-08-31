import argparse
import glob
import os

import cv2
import numpy as np

from funcs import create_dir


def main(args):
    create_dir(args.output_dir)
    assert os.path.exists(args.output_dir)
    mask_l = cv2.imread("mask_l_rendered.png")[:, :, 0]
    mask_r = cv2.imread("mask_r_rendered.png")[:, :, 0]

    for view in ["tgCloudPos_l", "tgCloudPos_r"]:
        mask = mask_l if view == "tgCloudPos_l" else mask_r
        root = os.path.join(args.root, view)
        assert os.path.exists(root)
        save_root = os.path.join(args.output_dir, view)
        create_dir(save_root)
        assert os.path.exists(save_root)

        for disp in glob.glob(os.path.join(root, r"*.npy")):
            disp_img = cv2.imread(disp)
            disp_img[mask == 0] = 0.3233
            np.save(os.path.join(save_root, disp.split("/")[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Mask the depth or disparity files with provided images for rendered data.")
    parser.add_argument("--root",
                        type=str,
                        default=r"rendered_disp")
    parser.add_argument("--output_dir",
                        type=str,
                        default=r"masked_rendered_disp")
    args = parser.parse_args()

    main(args)
