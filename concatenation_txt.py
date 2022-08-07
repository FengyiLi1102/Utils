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

    def write_lines(contents, index, f):
        f.write("{}{}".format(contents[index * 2], contents[index * 2 + 1]))

    def split(left_file, file_with_index, file_without_index, percentage):
        with open(left_file) as f:
            contents = f.readlines()
            n_frames = len(contents) / 2
            indexes = random.sample(range(0, int(n_frames - 1)), int(n_frames * percentage))

            with open(file_with_index, "w") as wif:
                with open(file_without_index, "w") as woif:
                    for index in np.arange(int(n_frames)):
                        if index in indexes:
                            write_lines(contents, index, wif)
                        else:
                            write_lines(contents, index, woif)

    print("Start concatenating...")
    funcs.create_dir(args.output_path)
    file_path = os.path.join(args.output_path, "concatenated_filenames.txt")
    train_file_path = os.path.join(args.output_path, "train_files.txt") if args.functional else ""
    concatenate_txt(args, ["splits/clouds/more_train.txt", "splits/clouds/more_val.txt"])

    val_file_path = ""
    test_file_path = ""

    if args.functional:
        if args.split_for_val:
            print("Start splitting the dateset for training and validating ...")
            # val_file_path = r"/vol/bitbucket/fl4718/monodepth2/splits/clouds/val_files.txt"
            val_file_path = os.path.join(args.output_path, r"val_files.txt")
            split(file_path, val_file_path, train_file_path, 0.2)

        if args.split_for_test:
            print("Start splitting the dataset for testing ...")
            test_file_path = os.path.join(args.output_path, r"test_files.txt")
            split(train_file_path, test_file_path, train_file_path, 0.125)

    if args.shuffle:
        if os.path.isfile(train_file_path):
            shuffle(train_file_path)
        if args.split_for_val and os.path.isfile(val_file_path):
            shuffle(val_file_path)
        if args.split_for_test and os.path.isfile(test_file_path):
            shuffle(test_file_path)


def concatenate_txt(args, file_path):
    with open(file_path, 'wb') as wfd:
        files = sorted(glob.glob(args.txt_path)) if type(file_path) == str else file_path
        print(f"Find {len(files)}. Save the output file in {args.output_path}.")
        for f in files:
            with open(f, 'rb') as fd:
                print(f"Open {f}")
                shutil.copyfileobj(fd, wfd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_path",
                        default="/vol/bitbucket/fl4718/monodepth2/splits/clouds/more",
                        type=str,
                        help="Path to txt files to be concatenated")
    parser.add_argument("--output_path",
                        default="/vol/bitbucket/fl4718/monodepth2/splits/clouds/",
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
    parser.add_argument("--split_for_test",
                        action="store_true",
                        dest="split_for_test",
                        default=False)
    parser.add_argument("--stereo",
                        action="store_true",
                        dest="stereo",
                        default=True)
    parser.add_argument("--functional",
                        action="store_true",
                        dest="functional",
                        default=False)

    args = parser.parse_args()

    main(args)
