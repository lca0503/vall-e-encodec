# Prepare Sox dataset

## Requirements

1. Install `Sox` package.
2. Install `requirements.txt` with the following command.

    ```
    pip install -r requirements.txt
    ```

## Usage

1. Download the LibriTTS dataset using `prepare_libritts.sh`:

    ```
    bash prepare_libritts.sh
    ```

2. Get the subset dataset from the LibriTTS dataset using `get_subset.py`:

    ```
    python3 get_subset.py -s train-clean-100 train-clean-360 train-other-500 -n -1 -o ./data/libritts_subset/train
    python3 get_subset.py -s dev-clean dev-other -n -1 -o ./data/libritts_subset/validation
    python3 get_subset.py -s test-clean test-other -n -1 -o ./data/libritts_subset/test
    ```

3. Assign effects to subset dataset using `assign_effects.py`:

    ```
    python3 assign_effects.py -d ./data/libritts_subset -s train -e bass treble chorus delay echo fade loudness repeat reverb reverse tempo vol pitch contrast
    python3 assign_effects.py -d ./data/libritts_subset -s validation -e bass treble chorus delay echo fade loudness repeat reverb reverse tempo vol pitch contrast
    python3 assign_effects.py -d ./data/libritts_subset -s test -e bass treble chorus delay echo fade loudness repeat reverb reverse tempo vol pitch contrast
    ```

4. Generate waveform with sox effects using `generate_waveform.sh`:

    ```
    bash generate_waveform.sh data/libritts_subset train
    bash generate_waveform.sh data/libritts_subset validation
    bash generate_waveform.sh data/libritts_subset test
    ```

5. Convert command to instruction using `command_to_instruction.py`:

    ```
    python3 command_to_instruction.py -d ./data/libritts_subset -s train -i ./data/instruction_candidates.json
    python3 command_to_instruction.py -d ./data/libritts_subset -s validation -i ./data/instruction_candidates.json
    python3 command_to_instruction.py -d ./data/libritts_subset -s test -i ./data/instruction_candidates.json
    ```

6. Convert waveform to encodec unit and upload subset dataset to huggingface using `waveform_to_unit.py`:

    ```
    python3 waveform_to_unit.py -d ./data/libritts_subset -s train validation test -o ./data/libritts_subset/soxdata_encodec
    ```

7. Upload dataset to huggingface. Login from the command line and run `upload_dataset.py`:

   ```
   huggingface-cli login
   python3 upload_dataset.py -d ./data/libritts_subset/soxdata_encodec -r lca0503/soxdata_encodec
   ```

## After running these commands, you should get ...
```
data/
├── instruction_candidates.json
├── libritts/
│   ├── train-clean-100/
│   ├── train-clean-360/
│   ├── train-other-500/
│   ├── dev-clean/
│   ├── dev-other/
│   ├── test-clean/
│   ├── test-other/
│   └── ...
└── libritts_subset/
    ├── soxdata_encodec/
    ├── train/
    │   ├── effects/
    │   │   ├── tempo.txt
    │   │   ├── bass.txt
    │   │   └── ...
    │   ├── source/
    │   │   ├── 000000.wav
    │   │   ├── 000001.wav
    │   │   └── ...
    │   ├── target/
    │   │   ├── 000000.wav
    │   │   ├── 000001.wav
    │   │   └── ...
    │   ├── command/
    │   │   ├── 000000.txt
    │   │   ├── 000001.txt
    │   │   └── ...
    │   ├── instruction/
    │   │   ├── 000000.txt
    │   │   ├── 000001.txt
    │   │   └── ...
    │   └── transcription/
    │       ├── 000000.txt
    │       ├── 000001.txt
    │       └── ...
    ├── validation/
    └── test/
```

## Download Dataset

Encodec Sox dataset is now available at [huggingface](https://huggingface.co/datasets/lca0503/soxdata_encodec).

| Dataset Split | Number of Instances in Split |
| ------------- | ---------------------------- |
| Train         | 354780                       |
| Validation    | 10349                        |
| Test          | 9957                         |

`effects: bass treble chorus delay echo fade loudness repeat reverb reverse tempo vol pitch contrast`