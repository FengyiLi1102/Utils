"""
Functions used in the Uitls.
"""

import glob
import os
import re
from datetime import datetime

import numpy as onp
from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from scipy.spatial.transform import Rotation as R


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        pass


def filenames_generator(n, n_local, l_vid_name, r_vid_name, args):
    output_path = os.path.join(args.filenames_output_dir, f"{args.W}_{args.H}_{datetime.today().strftime('%Y-%m-%d')}")
    create_dir(output_path)

    filenames_txt = open(os.path.join(output_path, l_vid_name + ".txt"), "w")

    # Shuffle the indexes or not
    indexes_list = onp.arange(n - n_local + 1, n)
    if args.shuffle:
        onp.random.shuffle(indexes_list)

    # Write the image names into the txt file for further training
    for i in indexes_list:
        filenames_txt.write(os.path.join("left", l_vid_name, f"img_{i}_left.png"))
        filenames_txt.write(" ")
        filenames_txt.write(os.path.join("right", r_vid_name, f"img_{i}_right.png"))
        filenames_txt.write("\n")

    filenames_txt.close()


def filenames_generator_mono(n, n_local, l_vid_name, r_vid_name, args):
    def txt_write_real(filenames_txt, i_img, l_vid_name, position):
        filenames_txt.write(os.path.join(position, l_vid_name))
        filenames_txt.write(" ")
        filenames_txt.write(str(i_img))
        filenames_txt.write(" ")
        filenames_txt.write(f"{position[0]}")
        filenames_txt.write("\n")

    def should_be_removed(index, n, n_local):
        if index == n - n_local + 1 or index == n:
            return True
        else:
            return False

    output_path = os.path.join(args.filenames_output_dir, f"{args.W}_{args.H}_{datetime.today().strftime('%Y-%m-%d')}")
    create_dir(output_path)

    filenames_txt = open(os.path.join(output_path, l_vid_name + ".txt"), "w")

    # Shuffle the indexes or not
    indexes_list = onp.arange(n - n_local + 1, n + 1)
    if args.shuffle:
        onp.random.shuffle(indexes_list)

    # Write the image names into the txt file for further training
    for i in indexes_list:
        if args.stereo or not should_be_removed(i, n, n_local):
            txt_write_real(filenames_txt, i, l_vid_name, "left")
            txt_write_real(filenames_txt, i, r_vid_name, "right")

    filenames_txt.close()


def filename_generator_rendered(n, n_local, output_path, file_name, shuffle=False):
    index_list = onp.arange(n - n_local + 1, n + 1)
    if shuffle:
        onp.random.shuffle(index_list)

    filenames_txt = open(os.path.join(output_path, file_name + ".txt"), "w")

    for i in index_list:
        txt_write_rendered(filenames_txt, i, "Left")
        txt_write_rendered(filenames_txt, i, "Right")

    filenames_txt.close()


def txt_write_rendered(filenames_txt, i_img, position):
    filenames_txt.write(position)
    filenames_txt.write(" ")
    filenames_txt.write(str("{:04d}".format(i_img)))
    filenames_txt.write(" ")
    filenames_txt.write(f"{position[0]}".lower())
    filenames_txt.write("\n")


def tif_to_png(img_dir, output_dir):
    create_dir(output_dir)
    i = 0
    for dir_path in glob.glob(os.path.join(img_dir, "*/")):
        new_output_dir = os.path.join(output_dir, dir_path.split("/")[-2])
        create_dir(new_output_dir)
        for img_path in sorted(glob.glob(os.path.join(dir_path, "*.tif"))):
            i += 1
            img_name = img_path.split("/")[-1].split(".")
            img_name[-1] = "PNG"
            img_name = ".".join(img_name)
            outfile = os.path.join(new_output_dir, img_name)
            im = Image.open(img_path)
            im.thumbnail(im.size)
            im.save(outfile, "png", quality=100)

            if i % 100 == 0:
                print(f"-> Convert {i} images ...")


def num_sort(input):
    return list(map(int, re.findall(r"\d+", input)))


def depth_to_disp(depth, b, f):
    return f * b / depth


def disp_to_depth(disp, b, f):
    return f * b / disp


def calculate_Tcw_Twc(view):
    if view == "l":
        tx, ty, tz, rx, ry, rz = -29.7387, 49.8636, -26.2908, 24.3174, -7.889, -0.872553  # left
    else:
        tx, ty, tz, rx, ry, rz = 22.2744, 30, -1.3696, 25.7777, -8.35386, -0.149347  # right

    r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    Twc = onp.eye(4)  # camera-to-world
    Twc[:3, :3] = r.as_matrix()
    Twc[:3, 3] = onp.array([tx, -ty, tz])
    Tcw = onp.linalg.inv(Twc)

    return Tcw, Twc


if __name__ == "__main__":
    filename_generator_rendered(1000, 1000, "", "rendered_train_data")
