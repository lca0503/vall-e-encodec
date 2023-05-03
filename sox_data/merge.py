import os
import random
import shutil
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd


def main(args):    
    random.seed(args.seed)
    
    assert os.path.isfile(f"{args.subset_dir}/transcriptions.tsv")
    assert os.path.isfile(f"{args.data_dir}/effect_to_instructions.json")

    transcriptions = pd.read_csv(f"{args.subset_dir}/transcriptions.tsv",
                                 sep='\t', header=None, index_col=False)
    transcriptions = transcriptions.rename(columns={0: "source_speech", 1: "transcription"})
    
    with open(f"{args.data_dir}/effect_to_instructions.json", "r") as f:
        effect_to_instructions = json.load(f)
        
    if os.path.isfile(f"{args.subset_dir}/tempo_commands.tsv"):
        tempo_commands = pd.read_csv(f"{args.subset_dir}/tempo_commands.tsv",
                                     sep='\t', header=None, index_col=False)
        tempo_commands = tempo_commands.rename(columns={0: "source_speech", 1: "target_speech", 2: "command", 3: "p1"})
        tempo_commands = tempo_commands.drop(["command"], axis=1)
        
        metadata_tempo = pd.merge(transcriptions, tempo_commands, on="source_speech", how = "inner")
        instructions_list = []
        for idx in range(len(metadata_tempo["source_speech"])):
            instruction = random.choice(effect_to_instructions["tempo"])
            instruction = instruction.replace("${file}", metadata_tempo["source_speech"][idx])
            instruction = instruction.replace("${newfile}", metadata_tempo["target_speech"][idx])
            instruction = instruction.replace("${p1}", str(metadata_tempo["p1"][idx]))
            instructions_list.append(instruction)
        metadata_tempo["instruction"] = instructions_list
        metadata_tempo = metadata_tempo.drop(["p1"], axis=1)
        
    if os.path.isfile(f"{args.subset_dir}/bass_commands.tsv"):
        bass_commands = pd.read_csv(f"{args.subset_dir}/bass_commands.tsv",
                                     sep='\t', header=None, index_col=False)
        bass_commands = bass_commands.rename(columns={0: "source_speech", 1: "target_speech", 2: "command", 3: "p1"})
        bass_commands = bass_commands.drop(["command"], axis=1)
        
        metadata_bass = pd.merge(transcriptions, bass_commands, on="source_speech", how = "inner")
        instructions_list = []
        for idx in range(len(metadata_bass["source_speech"])):
            instruction = random.choice(effect_to_instructions["bass"])
            instruction = instruction.replace("${file}", metadata_bass["source_speech"][idx])
            instruction = instruction.replace("${newfile}", metadata_bass["target_speech"][idx])
            instruction = instruction.replace("${p1}", str(metadata_bass["p1"][idx]))
            instructions_list.append(instruction)
        metadata_bass["instruction"] = instructions_list
        metadata_bass = metadata_bass.drop(["p1"], axis=1)

    metadata = pd.concat([metadata_tempo, metadata_bass], ignore_index=True)
    metadata.to_csv(args.output_file, sep="\t", header=False, index=False)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--n_sample", type=int, default=20000)
    parser.add_argument("-d", "--data_dir", type=str, default="./data")
    parser.add_argument("-i", "--subset_dir", type=str, default="./data/libritts_subset")
    parser.add_argument("-o", "--output_file", type=str, default="./data/libritts_subset/metadata.tsv")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
