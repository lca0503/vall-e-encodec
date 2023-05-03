# %%
import librosa
from encodec import EncodecModel
from encodec.utils import convert_audio
from pathlib import Path
import torchaudio
import torch
from tqdm import tqdm
from functools import partial
from datasets import Dataset
from time import time

# %%
instruction_dir = f"./data/libritts_subset/instruction"
transcription_dir = f"./data/libritts_subset/transcription"
source_audio_dir = f"./data/libritts_subset/source"
target_audio_dir = f"./data/libritts_subset/target"
source_audios = librosa.util.find_files(source_audio_dir, ext=["wav"])
target_audios = librosa.util.find_files(target_audio_dir, ext=["wav"])
file_ids = [Path(audio).stem for audio in source_audios]
test_audio = source_audios[0]
assert len(source_audios) == len(target_audios)
assert len(source_audios) > 0
assert len(target_audios) > 0
assert len(source_audios) == len(file_ids)

# %%
# 1. Instruction & Transcription(text)
def get_instruction(audio_id):
    instruction_path = Path(instruction_dir) / f"{audio_id}.txt"
    with open(instruction_path, "r") as f:
        instruction = f.read()
    return instruction

def get_transcription(audio_id):
    transcription_path = Path(transcription_dir) / f"{audio_id}.txt"
    with open(transcription_path, "r") as f:
        transcription = f.read()
    return transcription

# 2. Audio Path
def audio_path_from_id(audio_id, dir):
    return Path(dir) / f"{audio_id}.wav"
source_audio_path = partial(audio_path_from_id, dir=source_audio_dir)
target_audio_path = partial(audio_path_from_id, dir=target_audio_dir)

# 3. Audio Encodec Code
def convert_to_encodec_code(wav_path, model):
    wav, sr = torchaudio.load(wav_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
    return codes.squeeze(0).numpy()

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
encodec_24khz_8codebook = partial(convert_to_encodec_code, model=model)

# %%
def convert_to_datapoint_from_id(file_id):
    instruction = get_instruction(file_id)
    transcription = get_transcription(file_id)
    src_path = source_audio_path(file_id)
    tgt_path = target_audio_path(file_id)
    src_code = encodec_24khz_8codebook(src_path)
    tgt_code = encodec_24khz_8codebook(tgt_path)
    return file_id, instruction, transcription, src_code, tgt_code

# %%
dataset = {'file_id': [], 'instruction': [], 'transcription': [],}
for i in range(8):
    dataset[f'src_encodec_{i}'] = []
for i in range(8):
    dataset[f'tgt_encodec_{i}'] = []
print(dataset)

print(f"[INFO] There are len(file_ids) files to be processed.")
start = time()
for file_id in tqdm(file_ids):
    file_id, instruction, transcription, src_code, tgt_code = convert_to_datapoint_from_id(file_id)
    dataset['file_id'].append(file_id)
    dataset['instruction'].append(instruction)
    dataset['transcription'].append(transcription)
    for i in range(8):
        dataset[f'src_encodec_{i}'].append(list(src_code[i]))
        dataset[f'tgt_encodec_{i}'].append(list(tgt_code[i]))
        
print(f"[INFO] It takes {time() - start} seconds to process all files.")

# %%
SpeechChatGPT_dataset = audio_dataset = Dataset.from_dict(dataset)

# %%



