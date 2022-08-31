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
    directories = ["Left", "Right"] if not args.disp else ["tgCloudPos_l", "tgCloudPos_r"]
    ext = r"*.PNG" if not args.disp else r"*.npy"

    for view in directories:
        imgs_path = os.path.join(args.imgs_path, view)
        save_path = os.path.join(args.output_path, view)
        create_dir(save_path)
        print("Find rendered images in the {} directory ...".format(imgs_path))

        for i, img in enumerate(glob.glob(os.path.join(imgs_path, ext))):
            print("Start rectifying the rendered image --- {:04d}".format(i))
            # demo = np.reshape(np.load(img), (480, 640))
            # plt.imshow(demo, cmap='jet')
            # plt.colorbar()
            # plt.show()
            if view in ["Left", "tgCloudPos_l"]:
                rectify_rendered_frame(img, mapL1, mapL2, save_path, ext, args.min_disp)
            else:
                rectify_rendered_frame(img, mapR1, mapR2, save_path, ext, args.min_disp)


def rectify_rendered_frame(img, map1, map2, save_path, ext, min_disp):
    img_name = re.split(r'[/.]+', img)[-2]

    if ext == "*.PNG":
        calib_undistorted_unrectified_img = cv2.imread(img)
        calib_undistorted_rectified_img = cv2.remap(calib_undistorted_unrectified_img, map1, map2, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(save_path, "{}.{}".format(img_name, ext)), calib_undistorted_rectified_img)
    else:
        depth_img = np.load(img)
        depth_img = np.reshape(depth_img, (depth_img.shape[-2], depth_img.shape[-1]))
        rectified_depth_img = cv2.remap(depth_img, map1, map2, cv2.INTER_LINEAR)
        rectified_depth_img[rectified_depth_img <= 0.32344] = min_disp
        np.save(os.path.join(save_path, f"{img_name}.npy"), rectified_depth_img)

        # plt.imshow(rectified_depth_img, cmap='jet')
        # plt.colorbar()
        # plt.show()


def calculate_params(Tw_cl, Tw_cr):
    Tcr_cl = np.linalg.inv(Tw_cr) @ Tw_cl

    K = np.array([[f, 0., cx],
                  [0., f, cy],
                  [0., 0., 1.]], dtype=np.float64)

    dist_coeffs_l = np.zeros(5).astype(np.float64)
    dist_coeffs_r = np.zeros(5).astype(np.float64)
    rotation = Tcr_cl[:3, :3].astype(np.float64)
    translation = Tcr_cl[:3, -1].astype(np.float64)

    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(K, dist_coeffs_l, K, dist_coeffs_r, (640, 480), rotation,
                                                translation, alpha=-1)

    mapL1, mapL2 = cv2.initUndistortRectifyMap(K, dist_coeffs_l, RL, PL, (640, 480), cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(K, dist_coeffs_r, RR, PR, (640, 480), cv2.CV_32FC1)

    return mapL1, mapL2, mapR1, mapR2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Rectify rendered images and their corresponding disparity with provided camera poses.")
    parser.add_argument("--imgs_path",
                        type=str,
                        default=r"rendered_disp/")
    parser.add_argument("--output_path",
                        type=str,
                        default=r"rectified_rendered_disp/")
    parser.add_argument("--disp",
                        action="store_true",
                        dest="disp",
                        default=True)
    parser.add_argument("--min_disp",
                        type=float,
                        help="Maximum depth to set the depth value of the sky",
                        default=0.0)

    args = parser.parse_args()

    main(args)
