# Changes from the reference implementation

`core/` starts from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT licence, see `LICENSE`).
This file lists every change. The model architecture itself is unchanged.

## Telugu support

| File | Change |
|---|---|
| `text/telugu.py` (new) | Rule-based G2P: inherent vowel, matras, virama, geminates, place-assimilated anusvara, visarga. Number, ordinal and time expansion into Telugu words. |
| `text/symbols.py` | Adds the 46 Telugu phones and removes the unused pinyin set. |
| `preprocessor/align.py` (new) | No pretrained Telugu acoustic model exists, so MFA trains one on the corpus using the G2P lexicon. |
| `Telugu/config.yaml` | Single-speaker model (`multi_speaker: False`). |

## English frontend

| File | Change |
|---|---|
| `text/english.py` (new) | Runs G2P on the whole sentence with part-of-speech tags, so heteronyms such as "conduct" follow their grammar. The reference looked up a fixed lexicon first, which gave both uses the same pronunciation. |
| | Expands times, dates and symbols, and reads AM/PM as letters. Rejoins clitics (`'s`, `n't`). Adds a pronunciation for the code-mixed word "andaroo". |
| | `stock_phones()` keeps the reference behaviour for the comparison in the report (`--frontend stock`). |

## Data preparation

| File | Change |
|---|---|
| `preprocessor/prepare_data.py` (new) | Reads the Hugging Face parquet files directly. Filters clips, resamples to 22,050 Hz, splits with a fixed seed, and verifies the split. Replaces the per-dataset `prepare_align` scripts. |
| `preprocessor/preprocessor.py` | Uses fixed train/val id lists instead of a random `val_size`. Computes normalisation statistics on the training set only. Sorts file listings so speaker ids are reproducible. |

## Bugs fixed in the reference code

| File | Problem | Effect if left |
|---|---|---|
| `preprocessor/preprocessor.py` | Blank intervals in TextGrids were dropped. Current MFA versions write silence as blanks. | Pauses lose their frames, so durations stop matching the mel-spectrogram. |
| `text/__init__.py` | Unknown phones were silently removed. | Phone and duration sequences go out of step. Now raises an error. |
| `model/modules.py` | The energy predictor received `p_control` instead of `e_control`. | `--energy_control` had no effect and the pitch control scaled energy. |
| `audio/stft.py` | Hard-coded `.cuda()` calls. | Feature extraction crashed on machines without a GPU. |
| `preprocessor/preprocessor.py` | Audio was loaded at librosa's default rate and text with the system encoding. | Breaks with other configured rates and with non-ASCII text on some systems. |

## Speed and robustness

| File | Change |
|---|---|
| `model/modules.py` | Vectorised the length regulator with `repeat_interleave`. The original forced one GPU synchronisation per phoneme. Output was checked to be identical on 200 random cases. |
| `trainer.py` | Adds data-loading workers, a `--max_hours` limit for Kaggle's 12-hour sessions, checkpoint pruning, tracking of the best validation checkpoint, and skipping of out-of-memory batches. |
| `utils/model.py` | `check_vocoder_compatible()` stops the run if the mel settings differ from HiFi-GAN's config. |
| `preprocessor/align.py` | Runs MFA with `--clean`, because `mfa train` and `mfa align` share a database and keep overlapping alignments otherwise, which crashes the TextGrid export. Stops MFA if its working folder is unchanged for 45 minutes, because a crashed export worker leaves MFA waiting forever. |

## Library compatibility

| File | Change |
|---|---|
| `audio/stft.py`, `audio/audio_processing.py` | Keyword arguments required by librosa 0.10 and later. |
| `utils/model.py` | `weights_only=False` and `map_location` for PyTorch 2.6 and later. HiFi-GAN paths no longer depend on the working directory. |

## Layout

`train.py`, `synthesize.py` and the three YAML files per dataset became `trainer.py`, `synthesis.py` and one
`config.yaml` per language. `English/` and `Telugu/` contain small entry scripts that call this shared code,
so both models use the same implementation.
