import shutil
import argparse
import glob
import os
import funcs
import random


def shuffle(args):
    print("Start shuffling...")
    file_path = os.path.join(args.output_path, args.file_name)
    lines = open(file_path).readlines()
    random.shuffle(lines)
    open(file_path, "w").writelines(lines)


def main(args):
    print("Start concatenating...")
    funcs.create_dir(args.output_path)
    with open(os.path.join(args.output_path, args.file_name), 'wb') as wfd:
        files = glob.glob(args.txt_path)
        print(f"Find {len(files)}. Save the output file in {args.output_path}.")
        for f in files:
            with open(f, 'rb') as fd:
                print(f"Open {f}")
                shutil.copyfileobj(fd, wfd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("--txt_path",
                        required=True,
                        type=str,
                        help="Path to txt files to be concatenated")
    parser.add_argument("--output_path",
                        required=True,
                        type=str,
                        help="Output path for the generated file")
    parser.add_argument("--file_name",
                        type=str,
                        default="out_file.txt",
                        help="Output file name")
    parser.add_argument("--shuffle",
                        action="store_true",
                        dest="shuffle",
                        default=True)

    args = parser.parse_args()

    main(args)
    if args.shuffle:
        shuffle(args)
