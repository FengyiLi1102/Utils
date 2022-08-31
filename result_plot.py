import os

import matplotlib.pyplot as plt

from rectify_rendered import calculate_params

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
from funcs import calculate_Tcw_Twc
from funcs import disp_to_depth
import re

# Rendered settings
f = 530.227  # pixel
cx = 320
cy = 240

b = 61.0

# Compute transformation matrices
Tcl_w, Tw_cl = calculate_Tcw_Twc("l")
Tcr_w, Tw_cr = calculate_Tcw_Twc("r")
mapL1, mapL2, mapR1, mapR2 = calculate_params(Tw_cl, Tw_cr)


def mono_disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    Adapted from MonoDepth2 codes.
    GiHub: https://github.com/nianticlabs/monodepth2/blob/master/layers.py
    """
    min_disp = 1 / max_depth  # TODO: 1km - 10km
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


# world_points N x 3
def project_points(world_points, Tcw):
    N, cam_points = transform_coord(Tcw, world_points)

    img_points = np.zeros((N, 2))
    img_points[:, 0] = f * cam_points[:, 0] / cam_points[:, 2] + cx
    img_points[:, 1] = f * cam_points[:, 1] / cam_points[:, 2] + cy

    return img_points


def script():
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    # world_points N x 3
    def project_points(world_points, Tcw):
        N = world_points.shape[0]
        world_points_h = np.concatenate([world_points, np.ones((N, 1))], axis=1)
        cam_points = Tcw @ world_points_h.T
        cam_points = cam_points[:3, :].T

        img_points = np.zeros((N, 2))
        img_points[:, 0] = f * cam_points[:, 0] / cam_points[:, 2] + cx
        img_points[:, 1] = f * cam_points[:, 1] / cam_points[:, 2] + cy

        return img_points

    coord_img = cv2.imread(r"cloud_data_1k/tgCloudPos_r/depth_tgCloudPos_0134.exr",
                           cv2.IMREAD_UNCHANGED)[:, :, ::-1]

    coords = np.reshape(coord_img, [640 * 480, 3])
    coords[:, 1] = -coords[:, 1]
    np.savetxt('points.xyz', coords)

    f = 530.227
    cx = 320
    cy = 240

    tx, ty, tz, rx, ry, rz = 22.2744, 30, -1.3696, 25.7777, -8.35386, -0.149347
    r = R.from_euler('xyz', [rx, ry, rz], degrees=True)

    Twc = np.eye(4)
    Twc[:3, :3] = r.as_matrix()
    Twc[:3, 3] = np.array([tx, -ty, tz])
    Tcw = np.linalg.inv(Twc)

    img_coords = project_points(coords, Tcw)

    plt.subplot(1, 2, 1)
    plt.title('projected y coordinates')
    plt.imshow(np.reshape(img_coords, [480, 640, 2])[:, :, 1], vmin=0, vmax=480, cmap='jet')
    # plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title('projected x coordinates')
    plt.imshow(np.reshape(img_coords, [480, 640, 2])[:, :, 0], vmin=0, vmax=640, cmap='jet')
    # plt.colorbar()
    plt.show()


def transform_coord(T, points):
    N = points.shape[0]
    world_points_h = np.concatenate([points, np.ones((N, 1))], axis=1)
    cam_points = T @ world_points_h.T
    cam_points = cam_points[:3, :].T
    return N, cam_points


def load_rectify_depth(view, index):
    if view == "l":
        tx, ty, tz, rx, ry, rz = -29.7387, 49.8636, -26.2908, 24.3174, -7.889, -0.872553  # left
    else:
        tx, ty, tz, rx, ry, rz = 22.2744, 30, -1.3696, 25.7777, -8.35386, -0.149347  # right

    coord_img = cv2.imread(f"cloud_data_1k/tgCloudPos_{view}/depth_tgCloudPos_{index}.exr", cv2.IMREAD_UNCHANGED)[:, :,
                ::-1]  # R G B  -  x y z

    coords = np.reshape(coord_img, [640 * 480, 3])
    coords[:, 1] = -coords[:, 1]
    r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    Twc = np.eye(4)  # camera-to-world
    Twc[:3, :3] = r.as_matrix()
    Twc[:3, 3] = np.array([tx, -ty, tz])
    Tcw = np.linalg.inv(Twc)  # world-to-camera
    _, coords_c_frame = transform_coord(Tcw, coords)
    depth = coords_c_frame[:, -1]
    depth_img = np.reshape(depth, (480, 640))

    if view == "l":
        rectified_depth_img = cv2.remap(depth_img, mapL1, mapL2, cv2.INTER_LINEAR)
    else:
        rectified_depth_img = cv2.remap(depth_img, mapR1, mapR2, cv2.INTER_LINEAR)

    rectified_depth_img[rectified_depth_img <= 0] = 0

    return rectified_depth_img


def load_depth_mono(path_pred, index, view, title, model):
    rectified_depth_img = load_rectify_depth(view, index)
    max_depth_gt = np.max(rectified_depth_img)
    mask = cv2.imread(r"p35_l.png")[:, :, 0]

    """
    Ground truth depth
    """
    if title:
        plt.title(f"{index}_{view} Depth GT")
    plt.imshow(rectified_depth_img, cmap='jet')
    clb = plt.colorbar()
    clb.ax.set_title("Depth (m)")
    plt.savefig(f"{index}_depth_gt.png", bbox_inches='tight')
    plt.show()

    """
    Ground truth disparity
    """
    path = path_pred
    disp_img = np.load(path, fix_imports=True, encoding="latin1")
    path_gt = "rendered_disp/tgCloudPos_{0}/depth_tgCloudPos_{1}_disp.npy".format(view, index)
    disp_gt = np.load(path_gt)

    disp_gt = cv2.remap(disp_gt, mapL1, mapL2, cv2.INTER_LINEAR)
    if title:
        plt.title(f"{index}_{view} Disp GT")
    plt.imshow(disp_gt, cmap='jet')
    clb = plt.colorbar()
    clb.ax.set_title("Disparity (pixel)")
    plt.savefig(f"{index}_disp_gt.png", bbox_inches='tight')
    plt.show()

    """
    Predicted disparity
    """
    disp_img = np.reshape(disp_img, (480, 640))
    disp_img = disp_img * -1 if model else disp_img
    if title:
        plt.title(f"{path.split('/')[-1]} Disp Pred")
    plt.imshow(disp_img, cmap='jet')
    clb = plt.colorbar()
    clb.ax.set_title("Disparity (pixel)")
    plt.savefig(f"{index}_disp_pred.png", bbox_inches='tight')
    plt.show()

    """
    Predicted depth
    """

    path_index = re.split(r"[_/.]", path)[-3]
    disp_img[disp_img <= 0] = 0.1
    depth = disp_to_depth(disp_img, 61.0, 530.277) if model else mono_disp_to_depth(disp_img, 100, 10000)[-1]
    if title:
        plt.title(f"{path.split('/')[-1]} Depth Pred")

    if model:
        depth[mask == 0] = 0
        plt.imshow(depth, cmap='jet', vmax=8000)
    else:
        plt.imshow(1 / disp_img, cmap='jet')
    clb = plt.colorbar()
    clb.ax.set_title("Depth (m)")
    plt.savefig(f"{path_index}_depth_pred.png", bbox_inches='tight')
    plt.show()
    print("")

    rectified_depth_img[mask == 0] = 0
    plt.imshow(depth - rectified_depth_img, cmap='jet', vmax=2000)
    clb = plt.colorbar()
    clb.ax.set_title("Depth (m)")
    plt.savefig(f"{path_index}_diff.png", bbox_inches='tight')
    plt.show()

    rectified_depth_img[rectified_depth_img == 0] = 0.1
    depth[depth == 0] = 0.1
    print(compute_errors(rectified_depth_img, depth))


def plot_raw(path, title, index, view):
    f_c = 5.084752337001038995e+02
    disp_pred = np.load(path) * -1
    disp_pred[disp_pred <= 0] = 0.1

    if title:
        plt.title(f"{index}_{view} Disp GT")
    plt.imshow(disp_pred, cmap='jet')
    clb = plt.colorbar()
    clb.ax.set_title("Disparity (pixel)")
    plt.savefig(f"raw_{index}_disp_pred.png", bbox_inches='tight')
    plt.show()

    depth_pred = disp_to_depth(disp_pred, b, f_c)
    if title:
        plt.title(f"{path.split('/')[-1]} Depth Pred")
    plt.imshow(depth_pred, cmap='jet', vmax=7000)
    clb = plt.colorbar()
    clb.ax.set_title("Depth (m)")
    plt.savefig(f"raw_{index}_depth_pred.png", bbox_inches='tight')
    plt.show()


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    Adapted from: MonoDepth2 model (https://github.com/nianticlabs/monodepth2)
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


if __name__ == "__main__":
    # raw
    load_depth_mono(f"test_raw_l/img_526_left.npy",
                    "0363", "l", False, True)

    # rendered
    load_depth_mono(r"/Users/fyli/Documents/Msc_Computing/Individual_project/Models/RAFT-Stereo/demo_output/Left_80k"
                    r"/rgb_0363.npy",
                    "0363", "l",
                    True,
                    True)  # T for Raft and F for monodepth

    plot_raw(r"/Users/fyli/Documents/Msc_Computing/Individual_project/monodepth2/test_results_19/"
             r"img_249_left_disp_tl4_2021-09-29_13A.npy",
             True,
             "0249",
             "l")
