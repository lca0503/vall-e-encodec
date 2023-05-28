from argparse import ArgumentParser, Namespace

from datasets import DatasetDict


def main(args):
    Soxdataset = DatasetDict.load_from_disk(args.dataset_path)
    Soxdataset.push_to_hub(args.repo_name)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset_path", type=str, default="./data/libritts_subset/soxdata_encodec")
    parser.add_argument("-r", "--repo_name", type=str, default="lca0503/soxdata_encodec")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
