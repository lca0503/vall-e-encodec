import os
import random
import shutil
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def main(args):
    random.seed(args.seed)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/source").mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/transcription").mkdir(parents=True, exist_ok=True)
    path_to_transcription = defaultdict()
    
    with open(args.transcription_path, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.split('\t')
            assert len(line) == 2, line
            speech_path = line[0]
            transcription = line[1].strip()
            path_to_transcription[speech_path] = transcription

    data_subset = random.sample(path_to_transcription.items(), args.n_sample)
    
    for idx, (speech_path, transcription) in enumerate(tqdm(data_subset,
                                                            desc="Processing Subset", ascii=False, ncols=100)):
        new_file_idx = '{:06d}'.format(idx)
        new_speech_path = f"{args.output_dir}/source/{new_file_idx}.wav"
        new_transcription_path = f"{args.output_dir}/transcription/{new_file_idx}.txt"
        shutil.copyfile(speech_path, new_speech_path)
        with open(new_transcription_path, "w") as f:
            f.write(transcription)        
    
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--n_sample", type=int, default=20000)
    parser.add_argument("-t", "--transcription_path", type=str, default="./data/libritts_transcriptions.tsv")
    parser.add_argument("-o", "--output_dir", type=str, default="./data/libritts_subset")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
