import argparse
import glob
import os.path
import re

import cv2
import numpy as np

from funcs import create_dir, calculate_Tcw_Twc

f = 530.227  # pixel
cx = 320
cy = 240


def main(args):
    Tcl_w, Tw_cl = calculate_Tcw_Twc("l")
    Tcr_w, Tw_cr = calculate_Tcw_Twc("r")
    mapL1, mapL2, mapR1, mapR2 = calculate_params(Tw_cl, Tw_cr)

    create_dir(args.output_path)
    for view in ["Left", "Right"]:
        imgs_path = os.path.join(args.imgs_path, view)
        save_path = os.path.join(args.output_path, view)
        create_dir(save_path)
        print("Find rendered images in the {} directory ...".format(view))

        for i, img in enumerate(glob.glob(os.path.join(imgs_path, r"*.PNG"))):
            print("Start rectifying the rendered image --- {:04d}".format(i))
            if view == "Left":
                rectify_rendered_frame(img, mapL1, mapL2, save_path)
            else:
                rectify_rendered_frame(img, mapR1, mapR2, save_path)


def rectify_rendered_frame(img, map1, map2, save_path):
    img_name = re.split(r'[/.]+', img)[-2]
    calib_undistorted_unrectified_img = cv2.imread(img)
    calib_undistorted_rectified_img = cv2.remap(calib_undistorted_unrectified_img, map1, map2, cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(save_path, "{}.PNG".format(img_name)), calib_undistorted_rectified_img)


def calculate_params(Tw_cl, Tw_cr):
    Tcl_cr = np.linalg.inv(Tw_cl) @ Tw_cr

    K = np.array([[f, 0., cx],
                  [0., f, cy],
                  [0., 0., 1.]], dtype=np.float64)

    dist_coeffs_l = np.zeros(5).astype(np.float64)
    dist_coeffs_r = np.zeros(5).astype(np.float64)
    rotation = Tcl_cr[:3, :3].astype(np.float64)
    translation = Tcl_cr[:3, -1].astype(np.float64)

    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(K, dist_coeffs_l, K, dist_coeffs_r, (640, 480), rotation,
                                                translation, alpha=-1)

    mapL1, mapL2 = cv2.initUndistortRectifyMap(
        K, dist_coeffs_l, RL, PL, (640, 480),
        cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(
        K, dist_coeffs_r, RR, PR, (640, 480),
        cv2.CV_32FC1)

    return mapL1, mapL2, mapR1, mapR2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_path",
                        type=str,
                        default=r"rendered_train/")
    parser.add_argument("--output_path",
                        type=str,
                        default=r"rectified_rendered_data/")

    args = parser.parse_args()

    main(args)
