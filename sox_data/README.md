# Sox Data
## Usage
```
1. Download the LibriTTS dataset
bash prepare_libritts.sh

2. Get the transcriptions file of the LibriTTS dataset
python3 get_transcriptions.py

3. Get the subset from the LibriTTS dataset
python3 get_subset.py

4. Generate effect splits
python3 split_effects.py

5. Generate waveform with sox effects
bash waveform_generation.sh

6. Merge commands file and transcriptions file
python3 merge.py
```
## After running these commands, you should get ...
```
data/
├── effect_to_instructions.json
├── libritts/
│   ├── train-clean-100/
│   ├── train-clean-360/
│   ├── train-other-500/
│   └── ...
├── libritts_subset/
│   ├── commands.txt
│   ├── effect_splits/
│   │   ├── tempo.txt
│   │   ├── bass.txt
│   │   └── ...
│   ├── source/
│   │   ├── 000000.wav
│   │   ├── 000001.wav
│   │   └── ...
│   ├── target/
│   │   ├── 000000.wav
│   │   ├── 000001.wav
│   │   └── ...
│   ├── instruction/
│   │   ├── 000000.txt
│   │   ├── 000001.txt
│   │   └── ...
│   └── transcription/
│       ├── 000000.txt
│       ├── 000001.txt
│       └── ...
└── libritts_transcriptions.tsv
```