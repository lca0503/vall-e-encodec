import random
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from time import time

import librosa
import numpy as np
import torch
import torchaudio
from datasets import Dataset, DatasetDict
from encodec import EncodecModel
from encodec.utils import convert_audio
from tqdm import tqdm


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
# 1. Instruction & Transcription(text)
def get_instruction(audio_id, instruction_dir):
    instruction_path = Path(instruction_dir) / f"{audio_id}.txt"
    with open(instruction_path, "r") as f:
        instruction = f.read()
    return instruction


def get_transcription(audio_id, transcription_dir):
    transcription_path = Path(transcription_dir) / f"{audio_id}.txt"
    with open(transcription_path, "r") as f:
        transcription = f.read()
    return transcription


# 2. Audio Path
def audio_path_from_id(audio_id, dir):
    return Path(dir) / f"{audio_id}.wav"


def main(args):
    set_seed(args.seed)
    device = args.device

    # Setup dataset
    dataset_dict = {}

    # Setup model and codebook
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)

    for split in args.splits:
        print(f"[INFO] Process {split.upper()} split.")

        # Setup dataset
        dataset = {"file_id": [], "instruction": [], "transcription": [],}
        for i in range(8):
            dataset[f"src_encodec_{i}"] = []
        for i in range(8):
            dataset[f"tgt_encodec_{i}"] = []
        print(dataset)
            
        instruction_dir = f"{args.data_dir}/{split}/instruction"
        transcription_dir = f"{args.data_dir}/{split}/transcription"
        source_audio_dir = f"{args.data_dir}/{split}/source"
        target_audio_dir = f"{args.data_dir}/{split}/target"
        source_audios = librosa.util.find_files(source_audio_dir, ext=["wav"])
        target_audios = librosa.util.find_files(target_audio_dir, ext=["wav"])
        file_ids = [Path(audio).stem for audio in source_audios]
        assert len(source_audios) == len(target_audios)
        assert len(source_audios) > 0
        assert len(target_audios) > 0
        assert len(source_audios) == len(file_ids)

        source_audio_path = partial(audio_path_from_id, dir=source_audio_dir)
        target_audio_path = partial(audio_path_from_id, dir=target_audio_dir)

        print(f"[INFO] There are {len(file_ids)} files to be processed.")
        start = time()
        for idx, file_id in enumerate(tqdm(file_ids, desc="Converting", ascii=False, ncols=100)):
            instruction = get_instruction(file_id, instruction_dir)
            transcription = get_transcription(file_id, transcription_dir)
            src_path = source_audio_path(file_id)
            tgt_path = target_audio_path(file_id)

            wav, sr = torchaudio.load(src_path)
            wav = convert_audio(wav, sr, model.sample_rate, model.channels).unsqueeze(0)
    
            # Extract discrete codes from EnCodec
            with torch.no_grad():
                wav = wav.to(device)
                encoded_frames = model.encode(wav)
    
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
            src_code = list(codes.detach().cpu().squeeze(0).numpy())

            wav, sr = torchaudio.load(tgt_path)
            wav = convert_audio(wav, sr, model.sample_rate, model.channels).unsqueeze(0)
    
            # Extract discrete codes from EnCodec
            with torch.no_grad():
                wav = wav.to(device)
                encoded_frames = model.encode(wav)
    
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
            tgt_code = list(codes.detach().cpu().squeeze(0).numpy())
            
            dataset["file_id"].append(file_id)
            dataset["instruction"].append(instruction)
            dataset["transcription"].append(transcription)
            for i in range(8):
                dataset[f"src_encodec_{i}"].append(src_code[i])
                dataset[f"tgt_encodec_{i}"].append(tgt_code[i])

        print(f"[INFO] It takes {time() - start} seconds to process all files.")

        dataset = Dataset.from_dict(dataset)
        dataset_dict[split] = dataset

    Soxdataset = DatasetDict(dataset_dict)
    
    Soxdataset.save_to_disk(args.output_dir)

    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="./data/libritts_subset")
    parser.add_argument("-s", "--splits", nargs="*", default=["train", "validation", "test"])
    parser.add_argument("-o", "--output_dir", type=str, default="./data/libritts_subset/soxdata_encodec")
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=torch.device, default="cuda")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
