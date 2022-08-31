import argparse
import glob
import os

import cv2

import funcs


def video_generator(args):
    img_list = []
    height, width, layers = cv2.imread(os.path.join(args.frame_dir, r"img_1_left.png")).shape
    size = (width, height)

    print("Start...")
    frames_path = glob.glob(os.path.join(args.frame_dir, r"img_*_left.png"))
    frames_path.sort(key=funcs.num_sort)

    for frame in frames_path:
        img = cv2.imread(frame)
        img_list.append(img)

    video = cv2.VideoWriter(os.path.join(args.output_dir, args.video_name), 0,
                            args.frame_rate, size)

    for frame in img_list:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    print("Done...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate videos from given frames for real world video data.")
    parser.add_argument("--frame_dir",
                        type=str,
                        default="/Users/fyli/Documents/Msc_Computing/Individual_project/Models/RAFT-Stereo/demo_output/raw/20k_val/20000_val/tl4_2021-09-29_13A")
    parser.add_argument("--output_dir",
                        type=str,
                        default="")
    parser.add_argument("--video_name",
                        type=str,
                        default="output.avi")
    parser.add_argument("--frame_rate",
                        type=int,
                        default=31)

    args = parser.parse_args()

    video_generator(args)
