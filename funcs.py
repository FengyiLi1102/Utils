import os
from datetime import datetime

import cv2
import numpy as onp


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
        filenames_txt.write(path_generator("left", l_vid_name, f"img_{i}_left.png"))
        filenames_txt.write(" ")
        filenames_txt.write(path_generator("right", r_vid_name, f"img_{i}_right.png"))
        filenames_txt.write("\n")

    filenames_txt.close()


def filenames_generator_mono(n, n_local, l_vid_name, r_vid_name, args):
    output_path = os.path.join(args.filenames_output_dir, f"{args.W}_{args.H}_{datetime.today().strftime('%Y-%m-%d')}")
    create_dir(output_path)

    filenames_txt = open(os.path.join(output_path, l_vid_name + ".txt"), "w")

    # Shuffle the indexes or not
    indexes_list = onp.arange(n - n_local + 1, n)
    if args.shuffle:
        onp.random.shuffle(indexes_list)

    # Write the image names into the txt file for further training
    for i in indexes_list:
        txt_write(filenames_txt, i, l_vid_name, "left")
        txt_write(filenames_txt, i, r_vid_name, "right")

    filenames_txt.close()


def txt_write(filenames_txt, i_img, l_vid_name, position):
    filenames_txt.write(os.path.join(position, l_vid_name))
    filenames_txt.write(" ")
    filenames_txt.write(str(i_img))
    filenames_txt.write(" ")
    filenames_txt.write(f"{position[0]}")
    filenames_txt.write("\n")


def path_generator(direction, vid_name, img_name):
    return os.path.join(direction, vid_name, img_name)


def remove_timestamp(img, direction, args):
    if args.W == 512 and args.H == 256:
        if direction == "L":
            contours = onp.array([(166, 27), (169, 34), (247, 19), (245, 12)])
        else:
            contours = onp.array([(181, 9), (183, 12), (223, 5), (221, 2)])
    elif args.W == 640 and args.H == 480:
        if direction == "L":
            raise NotImplementedError
        else:
            raise NotImplementedError

    mask = onp.zeros(img.shape, dtype=onp.uint8)
    cv2.fillPoly(mask, pts=[contours], color=(255, 255, 255))
    mask_inv = cv2.bitwise_not(mask)

    # apply the mask
    masked_image = cv2.bitwise_and(img, mask_inv)

    # save the result
    return masked_image
