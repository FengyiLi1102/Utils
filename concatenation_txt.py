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
    file_path = os.path.join(args.output_path, args.file_name)
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
            print(contents[2])
            n_frames = len(contents) / 2
            print(n_frames)
            val_indexes = np.dot(random.sample(range(0, int(n_frames - 1)), int(n_frames * 0.2)), 2)

            with open(val_file_path, "w") as vf:
                for index in val_indexes:
                    vf.write("{}{}".format(contents[index], contents[index + 1]))

            with open(file_path, "w") as tf:
                for index, line in enumerate(contents):
                    if index not in val_indexes:
                        if index not in [0, 1, n_frames * 2 - 2, n_frames * 2 - 1]:
                            tf.write("{}".format(line))

    if args.shuffle:
        if os.path.isfile(file_path):
            shuffle(file_path)
        if args.split_for_val and os.path.isfile(val_file_path):
            shuffle(val_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_path",
                        default="datafile_names/640_480_2022-07-19/*.txt",
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

    args = parser.parse_args()

    main(args)
