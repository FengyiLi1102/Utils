import cv2
import glob
import argparse
import os


def video_generator(args):
    # frame_addr = r"/content/drive/MyDrive/RAFT-Stereo/demo_output/my_output/"
    # video_addr = r"/content/drive/MyDrive/output/"
    # video_name = r"colour_scaled_RAFT.avi"
    # frame_rate = 5

    img_list = []
    height, width, layers = cv2.imread(os.path.join(args.frame_dir, r"img_0.png")).shape
    size = (width, height)

    print("Start...")
    for frame in glob.glob(os.path.join(args.frame_dir, r"img_*.png")):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir",
                        type=str,
                        default="")
    parser.add_argument("--output_dir",
                        type=str,
                        default="")
    parser.add_argument("--video_name",
                        type=str,
                        default="output.avi")
    parser.add_argument("--frame_rate",
                        type=int,
                        default=5)

    args = parser.parse_args()

    video_generator(args);