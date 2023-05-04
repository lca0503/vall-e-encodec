import random
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from time import time

import librosa
import numpy as np
import torch
import torchaudio
from datasets import Dataset
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


# 3. Audio Encodec Code
def convert_to_encodec_code(wav_path, model, device):
    wav, sr = torchaudio.load(wav_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        wav = wav.to(device)
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
    return codes.cpu().squeeze(0).numpy()


def main(args):
    set_seed(args.seed)
    device = args.device
    
    instruction_dir = f"{args.data_dir}/instruction"
    transcription_dir = f"{args.data_dir}/transcription"
    source_audio_dir = f"{args.data_dir}/source"
    target_audio_dir = f"{args.data_dir}/target"
    source_audios = librosa.util.find_files(source_audio_dir, ext=["wav"])
    target_audios = librosa.util.find_files(target_audio_dir, ext=["wav"])
    file_ids = [Path(audio).stem for audio in source_audios]
    test_audio = source_audios[0]
    assert len(source_audios) == len(target_audios)
    assert len(source_audios) > 0
    assert len(target_audios) > 0
    assert len(source_audios) == len(file_ids)

    source_audio_path = partial(audio_path_from_id, dir=source_audio_dir)
    target_audio_path = partial(audio_path_from_id, dir=target_audio_dir)

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)
    
    encodec_24khz_8codebook = partial(convert_to_encodec_code, model=model, device=device)
    
    dataset = {'file_id': [], 'instruction': [], 'transcription': [],}
    for i in range(8):
        dataset[f'src_encodec_{i}'] = []
    for i in range(8):
        dataset[f'tgt_encodec_{i}'] = []
    print(dataset)

    print(f"[INFO] There are len(file_ids) files to be processed.")
    start = time()
    for file_id in tqdm(file_ids):
        instruction = get_instruction(file_id, instruction_dir)
        transcription = get_transcription(file_id, transcription_dir)
        src_path = source_audio_path(file_id)
        tgt_path = target_audio_path(file_id)
        src_code = encodec_24khz_8codebook(src_path)
        tgt_code = encodec_24khz_8codebook(tgt_path)
        
        dataset['file_id'].append(file_id)
        dataset['instruction'].append(instruction)
        dataset['transcription'].append(transcription)
        for i in range(8):
            dataset[f'src_encodec_{i}'].append(list(src_code[i]))
            dataset[f'tgt_encodec_{i}'].append(list(tgt_code[i]))
        
    print(f"[INFO] It takes {time() - start} seconds to process all files.")

    Soxdataset = Dataset.from_dict(dataset)
    Soxdataset.push_to_hub("lca0503/soxdata_small_encodec")

    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-d", "--data_dir", type=str, default="./data/libritts_subset")
    parser.add_argument("--device", type=torch.device, default="cuda")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)



