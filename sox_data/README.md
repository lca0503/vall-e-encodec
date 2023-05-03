# Sox Data
## Usage
```
1. Download the LibriTTS dataset.
bash prepare_libritts.sh

2. Get the transcriptions file of the LibriTTS dataset.
python3 get_transcriptions.py

3. Get the subset from the LibriTTS dataset.
python3 get_subset.py

4. Generate waveform with sox effects.
bash waveform_generation.sh

5. Merge commands file and transcriptions file. + Change commands into instructions.
python3 merge.py
```
## After running these commands, you should get ...
```
data/
    ├── effect_to_instructions.json
    ├── libritts_transcriptions.tsv
    ├── libritts/
    │   ├── train-clean-100/
    │   ├── train-clean-360/
    │   └──...
    └── libritts_subset/
        ├── transcriptions.tsv
        ├── bass_commands.tsv
        ├── tempo_commands.tsv
        ├── metadata.tsv
        ├── source
        │   ├── 000000.wav
        │   ├── 000001.wav
        │   └──...
        ├── target_bass
        │   ├── 000000.wav
        │   ├── 000001.wav
        │   └──...
        └── target_tempo
            ├── 000000.wav
            ├── 000001.wav
            └──...

```
metadata.tsv
```
<Source_speech_path>\t<Transcription>\t<Target_speech_path>\t<Instruction>
```