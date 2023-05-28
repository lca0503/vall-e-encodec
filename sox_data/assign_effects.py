import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm


def main(args):
    random.seed(args.seed)

    output_dir = f"{args.data_dir}/{args.split}/effects"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    todo = list(Path(f"{args.data_dir}/{args.split}/").rglob("*.wav"))
    num_effects = len(args.effects)
    files_per_effect = len(todo) // num_effects

    random.shuffle(todo)
    todo = [str(p) + "\n" for p in todo]
    
    start_idx = 0
    for effect in args.effects[:-1]:
        with open(f"{output_dir}/{effect}.txt", "w") as f:
            f.writelines(todo[start_idx:start_idx+files_per_effect])
        start_idx += files_per_effect
    with open(f"{output_dir}/{args.effects[-1]}.txt", "w") as f:
        f.writelines(todo[start_idx:])

        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="./data/libritts_subset")
    parser.add_argument("-s", "--split", type=str, default="train")
    parser.add_argument("-e", "--effects", nargs="*", default=["bass" "treble" "chorus" "delay" "echo" "fade" "loudness" "repeat" "reverb" "reverse" "tempo" "vol" "pitch" "contrast"])

    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
