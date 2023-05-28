import json
import os
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm


# sox -G <input.wav> <output.wav> bass <gain>
def command_bass(line, instruction_candidates):
    gain_ = line[1]
    instruction = random.choice(instruction_candidates[f"bass_gain{gain_}"])

    return instruction


# sox -G <input.wav> <output.wav> treble <gain>
def command_treble(line, instruction_candidates):
    gain_ = line[1]
    instruction = random.choice(instruction_candidates[f"treble_gain{gain_}"])
    
    return instruction


# sox -G <input.wav> <output.wav> chorus <gain-in> <gain-out> <delay> <decay> <speed> <depth> [-s|-t]
def command_chorus(line, instruction_candidates):
    gain_in_ = line[1]
    gain_out_ = line[2]
    delay_ = line[3]
    decay_ = line[4]
    speed_ = line[5]
    depth_ = line[6]
    modulation_ = line[7]
    instruction = random.choice(instruction_candidates[f"chorus_gain_in{gain_in_}"])

    return instruction


# sox -G <input.wav> <output.wav> delay <position>
def command_delay(line, instruction_candidates):
    position_ = line[1]
    instruction = random.choice(instruction_candidates["delay"])
    instruction = instruction.replace("{position}", position_)
    
    return instruction


# sox -G <input.wav> <output.wav> echo <gain-in> <gain-out> <delay> <decay>
def command_echo(line, instruction_candidates):
    gain_in_ = line[1]
    gain_out_ = line[2]
    delay_ = line[3]
    decay_ = line[4]
    instruction = random.choice(instruction_candidates[f"echo_delay{delay_}_decay{decay_}"])

    return instruction


# sox -G <input.wav> <output.wav> fade <type> <length>
def command_fade(line, instruction_candidates):
    type_ = line[1]
    length_ = line[2]
    instruction = random.choice(instruction_candidates["fade"])
    instruction = instruction.replace("{length}", length_)
    
    return instruction


# sox -G <input.wav> <output.wav> loudness <gain>
def command_loudness(line, instruction_candidates):
    gain_ = line[1]
    instruction = random.choice(instruction_candidates[f"loudness_gain{gain_}"])

    return instruction


# sox -G <input.wav> <output.wav> repeat <count>
def command_repeat(line, instruction_candidates):
    count_ = line[1]
    instruction = random.choice(instruction_candidates["repeat"])

    return instruction


# sox -G <input.wav> <output.wav> repeat <count>
def command_reverb(line, instruction_candidates):
    instruction = random.choice(instruction_candidates["reverb"])
    
    return instruction


# sox -G <input.wav> <output.wav> repeat <count>
def command_reverse(line, instruction_candidates):
    instruction = random.choice(instruction_candidates["reverse"])

    return instruction


# sox -G <input.wav> <output.wav> tempo <factor>
def command_tempo(line, instruction_candidates):
    factor_ = line[1]
    instruction = random.choice(instruction_candidates[f"tempo_factor{factor_}"])

    return instruction


# sox -G <input.wav> <output.wav> vol <gain>
def command_vol(line, instruction_candidates):
    gain_ = line[1]
    instruction = random.choice(instruction_candidates[f"vol_gain{gain_}"])

    return instruction


# sox -G <input.wav> <output.wav> pitch <cents>
def command_pitch(line, instruction_candidates):
    cents_ = line[1]
    instruction = random.choice(instruction_candidates[f"pitch_cents{cents_}"])

    return instruction


# sox -G <input.wav> <output.wav> contrast <amount>
def command_contrast(line, instruction_candidates):
    amount_ = line[1]
    instruction = random.choice(instruction_candidates[f"contrast_amount{amount_}"])
    
    return instruction


def main(args):
    random.seed(args.seed)
    
    output_dir = f"{args.data_dir}/{args.split}/instruction"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.instruction_candidates_path, "r") as f:
        instruction_candidates = json.load(f)

    todo = list(Path(f"{args.data_dir}/{args.split}/command").rglob("*.txt"))

    for command_path in tqdm(todo, desc="Select Instruction", ascii=False, ncols=100):
        with open(command_path, "r") as f:
            line = f.readline()
            line = line.strip()
            line = line.split("\t")
            effect = line[0]
            try:
                instruction = eval(f"command_{effect}(line, instruction_candidates)")
            except:
                raise ValueError("Effect not found")

            assert len(instruction) > 0
                        
        file_idx = os.path.basename(command_path).split(".")[0]
        with open(f"{output_dir}/{file_idx}.txt", "w") as f:
            f.write(instruction)        
                
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="./data/libritts_subset/")
    parser.add_argument("-s", "--split", type=str, default="train")
    parser.add_argument("-i", "--instruction_candidates_path", type=str, default="./data/instruction_candidates.json")

    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
