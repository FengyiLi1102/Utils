import argparse
import glob
import os
import random
import shutil

import numpy as np

import funcs


def main(args):
    def shuffle(file):
        print("Start shuffling...")
        lines = open(file).readlines()
        random.shuffle(lines)
        with open(file, "w") as f:
            f.writelines(lines)

    print("Start concatenating...")
    funcs.create_dir(args.output_path)
    file_path = os.path.join(args.output_path, "concatenated_temp_file.txt")
    train_file_path = os.path.join(args.output_path, "train_files.txt")
    with open(file_path, 'wb') as wfd:
        files = sorted(glob.glob(args.txt_path))
        print(f"Find {len(files)}. Save the output file in {args.output_path}.")
        for f in files:
            with open(f, 'rb') as fd:
                print(f"Open {f}")
                shutil.copyfileobj(fd, wfd)

    if args.split_for_val:
        print("Start splitting the dateset for training and validating ...")
        # val_file_path = r"/vol/bitbucket/fl4718/monodepth2/splits/clouds/val_files.txt"
        val_file_path = r"datafile_names/val_files.txt"
        with open(file_path) as f:
            contents = f.readlines()
            n_frames = len(contents) / 2
            val_indexes = random.sample(range(0, int(n_frames - 1)), int(n_frames * 0.2))
            print(val_indexes)

            with open(val_file_path, "w") as vf:
                with open(train_file_path, "w") as tf:
                    for index in np.arange(int(n_frames)):
                        if index in val_indexes:
                            write_lines(contents, index, vf)
                        else:
                            write_lines(contents, index, tf)

    if args.shuffle:
        if os.path.isfile(train_file_path):
            shuffle(train_file_path)
        if args.split_for_val and os.path.isfile(val_file_path):
            shuffle(val_file_path)


def write_lines(contents, index, f):
    f.write("{}{}".format(contents[index * 2], contents[index * 2 + 1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_path",
                        default="rendered_train_data.txt",
                        type=str,
                        help="Path to txt files to be concatenated")
    parser.add_argument("--output_path",
                        default="datafile_names",
                        type=str,
                        help="Output path for the generated file")
    parser.add_argument("--file_name",
                        type=str,
                        default="train_files.txt",
                        help="Output file name")
    parser.add_argument("--shuffle",
                        action="store_true",
                        dest="shuffle",
                        default=True)
    parser.add_argument("--mono",
                        action="store_true",
                        dest="mono",
                        default=True)
    parser.add_argument("--split_for_val",
                        action="store_true",
                        dest="split_for_val",
                        default=True)
    parser.add_argument("--stereo",
                        action="store_true",
                        dest="stereo",
                        default=True)

    args = parser.parse_args()

    main(args)
