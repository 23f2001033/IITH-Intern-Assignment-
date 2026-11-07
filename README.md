# FastSpeech 2 Text-to-Speech for English and Telugu

Two separately trained [FastSpeech 2](https://arxiv.org/abs/2006.04558) models, built on
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2):

| | English | Telugu |
|---|---|---|
| Dataset | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) |
| Speakers used | Multi-speaker, with speaker embeddings | One speaker (male, 48 kHz) |
| Text input | ARPAbet phonemes | Phonemes from a rule-based Telugu G2P |
| Alignment | MFA, pretrained `english_us_arpa` | MFA, acoustic model trained on the corpus |
| Vocoder | HiFi-GAN (universal), 22,050 Hz | HiFi-GAN (universal), 22,050 Hz |

The full analysis is in [Report/Technical_Report.pdf](Report/Technical_Report.pdf).
Every change to the reference code is listed in [core/CHANGES.md](core/CHANGES.md).

## Final models and results

| | English | Telugu |
|---|---|---|
| Checkpoint used | Step 60,000 (end of schedule) | Step 13,000 (lowest validation loss) |
| Voice | `spk_f0bb6d4e`, chosen on validation scores | The single dataset voice |
| Download | [`english_step60000.pth.tar`](https://huggingface.co/perceptron01/IIITH-FastSpeech2-English-Telugu) | [`telugu_step13000.pth.tar`](https://huggingface.co/perceptron01/IIITH-FastSpeech2-English-Telugu) |
| Validation WER (Whisper), 60 clips | 0.221, against 0.145 for vocoder-only resynthesis | Whisper unusable for Telugu (0.82 even on real speech) |
| Naturalness | UTMOS 2.63 (vocoder only: 2.65) | Native reviewers: clarity 5.0, naturalness 3.6 of 5 |

`Results/<Language>/` holds the test-sentence audio, the variance-adaptor plots (with the predicted values
as JSON) and the loss curves. Generated audio gets a small inference-time correction (longer final
vowel, quieter pauses, 0.3 s of silence at each end), explained in Section 7.4 of the report.

## Repository layout

```
English/, Telugu/   train.py, preprocess.py, inference.py, config.yaml, test_sentences.txt,
                    checkpoints/ (download link), assets/ (stats needed for inference)
core/               shared implementation, adapted from the reference repository
scripts/            loss plotting, speech evaluation and notebook generation
notebooks/          the Kaggle notebooks used for this work
Results/            loss curves, generated audio and variance-adaptor plots per language
Report/             technical report
```

## Setup

Python 3.11 was used for local testing. Montreal Forced Aligner is only needed for preprocessing.

```bash
pip install -r requirements.txt
conda install -c conda-forge montreal-forced-aligner   # or micromamba, as in notebooks/01_prepare_data.ipynb
hf auth login                                          # Svarah is gated: accept its terms first
```

## Reproducing the results

The runs were made on Kaggle with these notebooks, in order:

1. `notebooks/01_prepare_data.ipynb` (CPU).
2. `notebooks/02_train_and_generate.ipynb` (2 x T4, one language per GPU).
3. `notebooks/04_finish_english.ipynb`, which resumes English from step 54,153 to 60,000 after the 10-hour limit.
4. `notebooks/03_evaluate_speech.ipynb`, which scores the checkpoints and generates the final test sentences.

The same steps from a terminal:

```bash
hf download ai4bharat/Svarah --repo-type dataset --local-dir raw/svarah
hf download SPRINGLab/IndicTTS_Telugu --repo-type dataset --local-dir raw/telugu

# Preprocess: select and split clips, align with MFA, extract mel, pitch, energy and durations
python English/preprocess.py --corpus_path raw/svarah --work_dir work/English
python Telugu/preprocess.py  --corpus_path raw/telugu --work_dir work/Telugu

# Train (English stopped at the 10-hour limit at step 54,153 and was resumed to 60,000)
python English/train.py --work_dir work/English
python English/train.py --work_dir work/English --restore_step 54153 --anneal_steps 50000 55000 57500
python Telugu/train.py  --work_dir work/Telugu

# Score checkpoints on validation clips (Whisper WER/CER, UTMOS) and sweep English voices
python scripts/evaluate_tts.py --language English --work_dir work/English \
    --ckpt step60000=work/English/ckpt/60000.pth.tar --results_dir eval/English \
    --audio_dir eval_audio/English --sweep_speakers 12

# Generate audio and variance-adaptor plots
python English/inference.py --work_dir work/English --ckpt_file work/English/ckpt/60000.pth.tar --speaker spk_f0bb6d4e --results_dir Results/English
python Telugu/inference.py  --work_dir work/Telugu  --ckpt_file work/Telugu/ckpt/best.pth.tar  --results_dir Results/Telugu

# Loss curves
python scripts/plot_losses.py --log_dir work/English/log --out_dir Results/English --name English
python scripts/plot_losses.py --log_dir work/Telugu/log  --out_dir Results/Telugu  --name Telugu
```

To run inference with the released checkpoints, without preprocessing anything, download a checkpoint
(see `English/checkpoints/README.md` and `Telugu/checkpoints/README.md`) and use:

```bash
python English/inference.py --ckpt_file English/checkpoints/english_step60000.pth.tar --results_dir out/English
python English/inference.py --ckpt_file English/checkpoints/english_step60000.pth.tar --results_dir out/custom --text "Any sentence."
python Telugu/inference.py  --ckpt_file Telugu/checkpoints/telugu_step13000.pth.tar  --results_dir out/Telugu
```

The English voice defaults to `English/assets/target_speaker.txt` (`spk_f0bb6d4e`); pass `--speaker` to pick
another speaker from `English/assets/speakers.json`. The HiFi-GAN universal vocoder ships in the repository as
`core/hifigan/generator_universal.pth.tar.zip` and is unzipped automatically on first use.

## Data selection

Settings live in the `data` section of each `config.yaml`. The split uses a fixed seed (1234).

1. Keep clips of 1 to 15 seconds whose transcripts can be read from the text.
2. Resample everything to 22,050 Hz and normalise peak level.
3. Drop speaking-rate outliers. For English, also drop the noisiest 15% by estimated SNR.
4. Group clips that share a recording or a transcript, so a group always lands in one split.
5. Shuffle the groups and fill validation, then training, by cumulative duration.

Svarah has no speaker column, so speakers are identified from the combination of its metadata fields.
This gives 115 groups for its 117 speakers. Validation clips come from speakers also seen in training,
because a speaker embedding only learns from training clips. Clips, recordings and transcripts never
overlap between the splits, and `verify_split()` stops the run if they do.
