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
| `utils/tools.py` | Energy was the only target not cast to float32, and NumPy 2 saves it as float64 after normalisation. | Training crashed on the first GPU backward pass with `Found dtype Double but expected Float`. CPU runs do not show this. |
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

## Added after training

| File | Change |
|---|---|
| `trainer.py` | `--anneal_steps` overrides the learning-rate drop steps when resuming, used to finish the English schedule from step 54,153 to 60,000 within the remaining time. |
| `synthesis.py` | Inference-time correction of generated audio: a final vowel or sonorant shorter than 140 ms is lengthened to 140 ms and the decoder is rerun; vocoder output in predicted pauses is lowered to 10%; 5 ms fades and 0.3 s of silence are added at both ends. This compensates for training clips trimmed to the first and last phone (report, Section 7.4). |
| `synthesis.py` | Fixed the homograph comparison plot, where a reused loop variable overwrote the word shown in the title. |
| `../scripts/evaluate_tts.py` | New. Scores vocoder-only, teacher-forced and free synthesis on validation clips with Whisper (WER, CER) and UTMOS, with bootstrap intervals, and sweeps English voices. |
| `../scripts/plot_losses.py` | Training loss is smoothed with a centred rolling median, which does not shift the curve in steps or bend at learning-rate drops. |
