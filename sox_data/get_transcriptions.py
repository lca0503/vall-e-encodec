import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def main(args):
    todo = list(Path(args.data_dir).rglob("*.trans.tsv"))
    path_to_transcription = defaultdict()
    
    for transcription_file in tqdm(todo, desc="Loading", ascii=False, ncols=100):
        transcription_dir = os.path.dirname(transcription_file)
        with open(transcription_file, 'r') as f:
            content = f.readlines()
            for line in content:
                line = line.split('\t')
                speech_id = line[0]
                speech_path = f"{transcription_dir}/{speech_id}.wav"
                original_text = line[1]
                path_to_transcription[speech_path] = original_text
                
    with open(args.output_path, 'w') as f:
        for speech_path, transcription in path_to_transcription.items():
            f.write("\t".join([speech_path, transcription]))
            f.write("\n")
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="./data/libritts")
    parser.add_argument("-o", "--output_path", type=str, default="./data/libritts_transcriptions.tsv")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
