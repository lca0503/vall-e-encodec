import json
import os
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path


def main(args):
    random.seed(args.seed)
    
    output_dir = f"{args.data_dir}/{args.split}/instruction"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.effect_to_instructions_path, "r") as f:
        effect_to_instructions = json.load(f)

    todo = list(Path(f"{args.data_dir}/{args.split}/command").rglob("*.txt"))

    for command_path in todo:
        with open(command_path, "r") as f:
            line = f.readline()
            line = line.split("\t")
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
            
        file_idx = os.path.basename(source_speech_path).split(".")[0]
        with open(f"{output_dir}/{file_idx}.txt", "w") as f:
            f.write(instruction)        
                
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="./data/libritts_subset/")
    parser.add_argument("-s", "--split", type=str, default="train")
    parser.add_argument("-e", "--effect_to_instructions_path", type=str, default="./data/effect_to_instructions.json")

    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
