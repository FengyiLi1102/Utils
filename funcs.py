import glob
import os
from datetime import datetime

import cv2
import numpy as onp
from PIL import Image


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


def filename_generator_rendered(n, n_local, output_path, file_name):
    index_list = onp.arange(n - n_local + 1, n + 1)
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


def remove_timestamp(img, direction, args):
    if args.W == 512 and args.H == 256:
        if direction == "L":
            contours = onp.array([(166, 27), (169, 34), (247, 19), (245, 12)])
        else:
            contours = onp.array([(181, 9), (183, 12), (223, 5), (221, 2)])
    elif args.W == 640 and args.H == 480:
        if direction == "L":
            contours = onp.array([(208, 52), (212, 63), (308, 34), (305, 23)])
        else:
            contours = onp.array([(227, 17), (229, 24), (278, 10), (276, 4)])
    else:
        raise ValueError

    mask = onp.zeros(img.shape, dtype=onp.uint8)
    cv2.fillPoly(mask, pts=[contours], color=(255, 255, 255))
    mask_inv = cv2.bitwise_not(mask)

    # apply the mask
    masked_image = cv2.bitwise_and(img, mask_inv)

    # save the result
    return masked_image


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


if __name__ == "__main__":
    filename_generator_rendered(1000, 1000, "", "rendered_train_data")
