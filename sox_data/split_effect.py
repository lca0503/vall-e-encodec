import os
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm


def main(args):
    random.seed(args.seed)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    todo = list(Path(args.data_dir).rglob("*.wav"))
    num_splits = len(args.effects)
    files_per_effect = len(todo) // num_splits

    random.shuffle(todo)
    todo = [str(p) + "\n" for p in todo]
    
    start_idx = 0
    for effect in args.effects[:-1]:
        with open(f"{args.output_dir}/{effect}.txt", "w") as f:
            f.writelines(todo[start_idx:start_idx+files_per_effect])
        start_idx += files_per_effect
    with open(f"{args.output_dir}/{args.effects[-1]}.txt", "w") as f:
        f.writelines(todo[start_idx:])

        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-d", "--data_dir", type=str, default="./data/libritts_subset/source")
    parser.add_argument("-o", "--output_dir", type=str, default="./data/libritts_subset/effect_splits")
    parser.add_argument("-e", "--effects", nargs='*', default=["bass", "tempo"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
