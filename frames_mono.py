import argparse
import glob
import re

from funcs import filenames_generator_mono


def main(args):
    n = 989
    n_local = 989
    for (l_path, r_path) in zip(sorted(glob.glob(r"raw_videos/Left/*.mp4")),
                                sorted(glob.glob(r"raw_videos/Right/*.mp4"))):
        l_name = re.split(r"[/.]+", l_path)[-2]
        r_name = re.split(r"[/.]+", r_path)[-2]
        l_date = l_name.split("_")[1:]
        r_date = r_name.split("_")[1:]

        assert (l_date == r_date)
        filenames_generator_mono(n, n_local, l_name, r_name, args)
        n += n_local


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--H",
                        type=int,
                        help="Height of the frame",
                        default=480)
    parser.add_argument("--W",
                        type=int,
                        help="Width of the frame",
                        default=640)
    parser.add_argument("--filenames_output_dir",
                        type=str,
                        help="Path to store the dataset filenames in txt fime",
                        default="datafile_names/")
    parser.add_argument("--shuffle",
                        dest="shuffle",
                        action="store_true",
                        help="Shuffle the frame indexes or not",
                        default=True)
    parser.add_argument("--stereo",
                        dest="stereo",
                        action="store_true",
                        help="If set, generate filename texts for stereo training",
                        default=True)

    args = parser.parse_args()

    main(args)
