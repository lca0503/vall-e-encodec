import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def main(args):
    todo = list(Path(args.data_dir).rglob("*.original.txt"))
    path_to_transcription = defaultdict()
    
    for transcription_path in tqdm(todo, desc="Loading", ascii=False, ncols=100):
        transcription_dir = os.path.dirname(transcription_path)
        speech_id = str(os.path.basename(transcription_path)).split('.')[0]
        speech_path = f"{transcription_dir}/{speech_id}.wav"        
        assert os.path.isfile(speech_path)
        with open(transcription_path, 'r') as f:
            transcription = ""
            content = f.readlines()
            for line in content:
                transcription += str(line)
                path_to_transcription[speech_path] = transcription
                
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
