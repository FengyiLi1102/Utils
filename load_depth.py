import argparse
import glob
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
import re
from funcs import create_dir, depth_to_disp, calculate_Tcw_Twc

file_dir = os.path.dirname(__file__)


def load_rendered_depth(args):
    create_dir(args.output_path)
    for view in ["tgCloudPos_l", "tgCloudPos_r"]:
        view_path = os.path.join(args.exr_path, view)
        disp_dir = os.path.join(args.output_path, view)
        create_dir(disp_dir)

        Tcw, Twc = calculate_Tcw_Twc(view.split("_")[-1])

        print(f"Start to extract ground truth depth from {view} directory ...")
        n = len(glob.glob(os.path.join(view_path, "*.exr")))

        for i, exr in enumerate(glob.glob(os.path.join(view_path, "*.exr"))):
            print("-> Extract {} depth".format(i + 1))
            coord_img = cv2.imread(exr, cv2.IMREAD_UNCHANGED)[:, :, ::-1]  # R G B  -  x y z
            coords = np.reshape(coord_img, [640 * 480, 3])
            coords[:, 1] = -coords[:, 1]
            index = re.split(r'[/.]+', exr)[-2]
            np.savetxt(os.path.join(disp_dir, f"{index}_points.xyz"), coords)

            _, coords_c_frame = transform_coord(Tcw, coords)
            depth = coords_c_frame[:, -1]
            depth_img = np.reshape(depth, (480, 640))
            depth_img[depth_img < args.min_depth] = 100000
            """
            Baseline: norm of the displacement vector between two cameras (b = 61.0)
            """
            disp_img = depth_to_disp(depth_img, 61.0, 530.227)
            disp_img = np.reshape(disp_img, (480, 640))
            np.save(os.path.join(os.path.join(args.output_path, view), f"{index}_disp.npy"), disp_img)


def transform_coord(T, points):
    N = points.shape[0]
    world_points_h = np.concatenate([points, np.ones((N, 1))], axis=1)
    cam_points = T @ world_points_h.T
    cam_points = cam_points[:3, :].T
    return N, cam_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load depth from the rendered data in the form of exr.")
    parser.add_argument("--exr_path",
                        type=str,
                        help="Path to the directory containing all exr files.",
                        default=os.path.join(file_dir, "cloud_data_1k"))
    parser.add_argument("--output_path",
                        type=str,
                        help="Path to the directory containing all generated disparity file.",
                        default=os.path.join(file_dir, "rendered_disp"))
    parser.add_argument("--min_depth",
                        type=int,
                        default=300)
    parser.add_argument("--max_depth",
                        type=int,
                        default=10000)
    args = parser.parse_args()
    load_rendered_depth(args)
