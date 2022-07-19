import argparse
import glob
import os
from datetime import datetime
from shutil import make_archive

import cv2
from tqdm import tqdm

import calibration
import funcs


# def load_calibration_info(calib_path):
#     print("Loading calibration information...")
#     text_file_paths = os.path.join(calib_path, r"*.txt")

#     if len(glob.glob(text_file_paths)) == 0:
#         calibration.calibration()

#     calib_dict = {}
#     for text_file_path in glob.glob(text_file_paths):
#         calibration_param = re.split(r"[/\_\.]+", text_file_path)[-2]
#         print(f"Load {calibration_param} from {text_file_path.split('/')[-1]}")
#         calib_dict[calibration_param] = onp.array(onp.loadtxt(text_file_path, delimiter=" ", unpack=False))

#     print("Calibration loading finished\n")
#     return calib_dict


# Generate the rectified frames
def main(args):
    # Load calibration information
    # calib_data = load_calibration_info(args.calibration_path)
    mapL1, mapL2, mapR1, mapR2 = calibration.main()

    frame_size = (args.W, args.H)

    left_videos = sorted(glob.glob(args.video_L, recursive=True))
    right_videos = sorted(glob.glob(args.video_R, recursive=True))
    n_videos = len(left_videos)
    print(f"Found {n_videos} video(s).")

    output_dir_name = f"{args.W}_{args.H}_{datetime.today().strftime('%Y-%m-%d')}"
    frames_output_dir = os.path.join(args.frames_output_dir, output_dir_name)
    funcs.create_dir(frames_output_dir)
    print(f"Saving frames to {frames_output_dir}.")

    l_output_path = os.path.join(frames_output_dir, "left")
    r_output_path = os.path.join(frames_output_dir, "right")
    funcs.create_dir(l_output_path)
    funcs.create_dir(r_output_path)

    n = 0

    for (cap_left_path, cap_right_path) in tqdm(list(zip(left_videos, right_videos))):
        n_local = 0  # count number of frames finally generated in the current video
        myCapL = cv2.VideoCapture(cap_left_path)
        myCapR = cv2.VideoCapture(cap_right_path)

        left_video_name = cap_left_path.split("/")[-1]
        right_video_name = cap_right_path.split("/")[-1]

        l_vid_name = left_video_name.split(".")[0]
        r_vid_name = right_video_name.split(".")[0]

        assert (left_video_name.split("_")[1] == right_video_name.split("_")[1])

        print('\n')
        print(f"Extract frames from left video: {left_video_name} and from right video: {right_video_name}.")

        left_output_path = os.path.join(frames_output_dir,
                                        "left",
                                        l_vid_name)
        funcs.create_dir(left_output_path)

        right_output_path = os.path.join(frames_output_dir,
                                         "right",
                                         r_vid_name)
        funcs.create_dir(right_output_path)

        while myCapL.isOpened():
            ret, frame_L = myCapL.read()
            if not ret:
                print(f"Finish extracting {left_video_name}")
                print(f"Total {n_local} images \n")
                break
            _, frame_R = myCapR.read()

            n += 1  # Total number of frames generated
            n_local += 1  # Total number of frames in the current video

            # Set the frame rate
            # CAP_PROP_POS_MSEC: Current position of the video file in milliseconds
            myCapL.set(cv2.CAP_PROP_POS_MSEC, args.frame_per_sec * n_local)
            myCapR.set(cv2.CAP_PROP_POS_MSEC, args.frame_per_sec * n_local)

            frame_L = cv2.resize(frame_L, (640, 480))
            frame_R = cv2.resize(frame_R, (640, 480))

            # Rectified
            if args.rectified:
                frame_L = cv2.remap(frame_L, mapL1, mapL2, cv2.INTER_LINEAR)
                frame_R = cv2.remap(frame_R, mapR1, mapR2, cv2.INTER_LINEAR)

            frame_L = cv2.resize(frame_L, frame_size)
            frame_R = cv2.resize(frame_R, frame_size)

            # Remove the timestamp on frames
            frame_L = funcs.remove_timestamp(frame_L, "L", args)
            frame_R = funcs.remove_timestamp(frame_R, "R", args)

            # Save images
            left_img_name = f"img_{n}_left.png"
            right_img_name = f"img_{n}_right.png"
            left_img_output_path = os.path.join(left_output_path, left_img_name)
            right_img_output_path = os.path.join(right_output_path, right_img_name)
            cv2.imwrite(left_img_output_path, frame_L)
            cv2.imwrite(right_img_output_path, frame_R)

            # # Remove the timestamp on frames
            # l_img = Image.open(left_img_output_path).convert("RGBA")
            # r_img = Image.open(right_img_output_path).convert("RGBA")
            # funcs.remove_timestamp(l_img, "L", left_img_output_path)
            # funcs.remove_timestamp(r_img, "R", right_img_output_path)

            if n % 50 == 0:
                print(f"Finish {n} frame...")

        myCapL.release()
        myCapR.release()
        cv2.destroyAllWindows()

        print(f"Total {n} images are extracted from {n_videos} videos.")

        # Generate filename file for training and validation
        if args.model == "Unsup":
            funcs.filenames_generator(n, n_local, l_vid_name, r_vid_name, args)
        elif args.model == "mono":
            funcs.filenames_generator_mono(n, n_local, l_vid_name, r_vid_name, args)
        else:
            raise ValueError

    # Zip all generated frames based on their output directories
    print("Zipping frame directories...")
    make_archive(os.path.join(args.frames_output_dir, output_dir_name),
                 "zip",
                 root_dir=args.frames_output_dir,
                 base_dir=output_dir_name)

    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_L",
                        help="Path to all the videos from left view required to"
                             " be extracted frames",
                        # default="/Users/fyli/Documents/Msc Computing/Individual_project/Utils/raw_videos/Left/*.mp4",
                        default="raw_videos/Left/*.mp4")
    parser.add_argument("--video_R",
                        help="Path to all the videos from right view required to"
                             "be extracted frames",
                        # default="/Users/fyli/Documents/Msc Computing/Individual_project/Utils/raw_videos/Right/*.mp4,
                        default="raw_videos/Right/*.mp4")
    parser.add_argument("--H",
                        type=int,
                        help="Height of the frame",
                        default=480)
    parser.add_argument("--W",
                        type=int,
                        help="Width of the frame",
                        default=640)
    parser.add_argument("--frame_per_sec",
                        type=int,
                        help="Number of frames extracted per second",
                        default=29)  # Default: generate 1000 frames
    parser.add_argument("--frames_output_dir",
                        help="Path to where frames from left view to be stored",
                        default="frames_output/")
    parser.add_argument("--filenames_output_dir",
                        type=str,
                        help="Path to store the dataset filenames in txt fime",
                        default="datafile_names/")
    parser.add_argument("--shuffle",
                        dest="shuffle",
                        action="store_true",
                        help="Shuffle the frame indexes or not",
                        default=True)
    parser.add_argument("--rectified",
                        dest="rectified",
                        action="store_true",
                        help="Rectify the frames generated from the video",
                        default=True)
    parser.add_argument("--zip",
                        dest="zip",
                        action="store_true",
                        help="Zip frames",
                        default=False)
    parser.add_argument("--model",
                        type=str,
                        help="Name of the model used for training",
                        choices=["Unsup", "mono"],
                        default="mono")
    parser.add_argument("--stereo",
                        dest="stereo",
                        action="store_true",
                        help="If set, generate filename texts for stereo training",
                        default=False)

    args = parser.parse_args()

    main(args)
