import os
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json


def main(args):    
    random.seed(args.seed)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{args.data_dir}/effect_to_instructions.json", "r") as f:
        effect_to_instructions = json.load(f)
    
    with open(f"{args.commands_path}", "r") as f:
        content = f.readlines()
        for line in content:
            line = line.split('\t')
            source_speech_path = line[0]
            target_speech_path = line[1]
            effect = line[2]
            instruction = ""
            if effect == "tempo":
                p1 = line[3].strip()
                instruction = random.choice(effect_to_instructions["tempo"])
                instruction = instruction.replace("${file}", source_speech_path)
                instruction = instruction.replace("${newfile}", target_speech_path)
                instruction = instruction.replace("${p1}", p1)
            elif effect == "bass":
                p1 = line[3].strip()
                instruction = random.choice(effect_to_instructions["bass"])
                instruction = instruction.replace("${file}", source_speech_path)
                instruction = instruction.replace("${newfile}", target_speech_path)
                instruction = instruction.replace("${p1}", p1)
            else:
                continue
            
            file_idx = os.path.basename(source_speech_path).split('.')[0]
            with open(f"{args.output_dir}/{file_idx}.txt", "w") as f:
                f.write(instruction)        
                
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--n_sample", type=int, default=20000)
    parser.add_argument("-d", "--data_dir", type=str, default="./data")
    parser.add_argument("-c", "--commands_path", type=str, default="./data/libritts_subset/commands.txt")
    parser.add_argument("-o", "--output_dir", type=str, default="./data/libritts_subset/instruction")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
