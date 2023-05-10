import random
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm


def main(args):
    random.seed(args.seed)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/source").mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/transcription").mkdir(parents=True, exist_ok=True)

    todo = []
    for split in args.splits: 
        todo += list(Path(f"{args.data_dir}/{split}").rglob("*.wav"))

    if args.n_sample == -1:
        data_subset = todo
    else:
        data_subset = random.sample(todo, args.n_sample)

    for idx, speech_path in enumerate(tqdm(data_subset, desc="Processing Subset", ascii=False, ncols=100)):
        transcription_path = str(speech_path).replace(".wav", ".original.txt")
        new_file_idx = "{:06d}".format(idx)
        new_speech_path = f"{args.output_dir}/source/{new_file_idx}.wav"
        new_transcription_path = f"{args.output_dir}/transcription/{new_file_idx}.txt"
        shutil.copyfile(speech_path, new_speech_path)
        shutil.copyfile(transcription_path, new_transcription_path)
    
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="./data/libritts")
    parser.add_argument("-s", "--splits", nargs="*", default=["train-clean-100", "train-clean-360", "train-other-500"])
    parser.add_argument("-n", "--n_sample", type=int, default=20000)
    parser.add_argument("-o", "--output_dir", type=str, default="./data/libritts_subset/train")
    
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
