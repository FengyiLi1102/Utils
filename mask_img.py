import argparse
import glob
import os

import cv2

from funcs import create_dir


def main(args):
    mask_l = cv2.imread(r"mask_1.png")
    mask_r = cv2.imread(r"mask_2.png")
    mask_r_test = cv2.imread(r"mask_r_test.png")
    n = 0

    print("-> Start masking ...")
    for direction_folder in glob.glob(os.path.join(args.root_dir, "*/")):
        masked_dir = creat_dir_mono(args, args.output_name)
        print(f"-> Create folder {masked_dir}")
        folder_path = os.path.join(direction_folder, "*/")
        print(f"-> Open folder {direction_folder} ...")
        masked_direction_dir = os.path.join(masked_dir, direction_folder.split("/")[-2])

        for video_folder in glob.glob(folder_path):
            masked_video_dir = os.path.join(masked_direction_dir, video_folder.split("/")[-2])
            create_dir(masked_video_dir)
            print(f"-> Create {masked_video_dir}")
            image_paths = os.path.join(video_folder, "*.png")
            print(f"-> Open folder {video_folder}")

            for frame_path in glob.glob(image_paths):
                img = cv2.imread(frame_path)
                mask = mask_l if direction_folder.split("/")[-1] == "left" else mask_r_test
                masked_img = cv2.bitwise_and(mask, img)
                path_list = frame_path.split("/")
                path_list[1] = args.output_name
                create_dir("/".join(path_list[:-2]))
                masked_img_path = "/".join(path_list)
                # cv2.imshow("test", masked_img)
                # cv2.waitKey(0)

                cv2.imwrite(masked_img_path, masked_img)
                n += 1

                if n % 100 == 0:
                    print(f"-> Finish masking {n} images")

    print(f"Finish masking all {n} images.")


def creat_dir_mono(args, dir_name):
    new_root_path = args.root_dir.split("/")
    new_root_path[-1] = dir_name
    new_root_path = "/".join(new_root_path)
    create_dir(new_root_path)

    return new_root_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",
                        type=str,
                        default="frames_output/640_480_2022-07-18")
    parser.add_argument("--output_name",
                        type=str,
                        default=r"640_480_train")
    opts = parser.parse_args()
    main(opts)
