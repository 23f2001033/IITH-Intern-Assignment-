"""Score generated speech on validation sentences to find where quality is lost.

Each clip is produced in up to three ways:

    vocoder            true mel-spectrogram -> HiFi-GAN
    teacher:<ckpt>     model given the true durations, pitch and energy -> HiFi-GAN
    free:<ckpt>        model predicting everything itself -> HiFi-GAN

Comparing them separates vocoder errors from acoustic model and predictor errors.
Whisper transcribes every clip for word and character error rates, and UTMOS
predicts naturalness. Test sentences are never used, so choices made from these
scores do not tune anything on the test set.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import librosa
import numpy as np
import torch
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

from config import load_config  # noqa: E402
from text import text_to_sequence  # noqa: E402
from utils.model import check_vocoder_compatible, get_model, get_vocoder, vocoder_infer  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WHISPER = "openai/whisper-large-v3-turbo"
WHISPER_LANGUAGE = {"English": "en", "Telugu": "te"}
LISTEN_CLIPS = 6


def normalise(text):
    """Lower-case and keep letters, vowel signs and digits, so scoring ignores punctuation."""
    kept = [c if unicodedata.category(c)[0] in "LMN" else " " for c in text.lower()]
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def error_rate(reference, hypothesis):
    return edit_distance(reference, hypothesis) / max(len(reference), 1)


def read_filelist(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            basename, speaker, phones, text = line.rstrip("\n").split("|", 3)
            rows.append({"id": basename, "speaker": speaker, "phones": phones, "text": text})
    return rows


def feature(folder, kind, row):
    return np.load(os.path.join(folder, kind, "{}-{}-{}.npy".format(row["speaker"], kind, row["id"])))


def synthesize(model, vocoder, configs, phones, speaker, targets=None):
    preprocess_config, model_config, _ = configs
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    sequence = np.array(text_to_sequence(phones, cleaners))
    inputs = [
        torch.tensor([speaker]).long().to(device),
        torch.from_numpy(sequence[None, :]).long().to(device),
        torch.tensor([len(sequence)]).to(device),
        len(sequence),
    ]
    if targets is not None:
        frames = int(targets["duration"].sum())
        inputs += [
            None,
            torch.tensor([frames]).to(device),
            frames,
            torch.from_numpy(targets["pitch"][None, :]).float().to(device),
            torch.from_numpy(targets["energy"][None, :]).float().to(device),
            torch.from_numpy(targets["duration"][None, :]).long().to(device),
        ]
    with torch.no_grad():
        output = model(*inputs)
    hop = preprocess_config["preprocessing"]["stft"]["hop_length"]
    return vocoder_infer(
        output[1].transpose(1, 2), vocoder, model_config, preprocess_config, lengths=output[9] * hop
    )[0]


def generate(args, configs, checkpoints):
    preprocess_config, model_config, _ = configs
    folder = preprocess_config["path"]["preprocessed_path"]
    rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    with open(os.path.join(folder, "speakers.json"), encoding="utf-8") as f:
        speakers = json.load(f)
    val = read_filelist(os.path.join(folder, "val.txt"))
    random.Random(1234).shuffle(val)
    clips = val[: args.clips]
    vocoder = get_vocoder(model_config, device)
    manifest = []

    def save(condition, row, wav, speaker=None):
        folder_out = os.path.join(args.audio_dir, condition.replace(":", "_").replace("@", "_"))
        os.makedirs(folder_out, exist_ok=True)
        path = os.path.join(folder_out, row["id"] + ".wav")
        wavfile.write(path, rate, wav)
        manifest.append({"condition": condition, "id": row["id"], "path": path, "text": row["text"],
                         "speaker": speaker or row["speaker"]})

    for row in clips:
        mel = torch.from_numpy(feature(folder, "mel", row).T[None]).float().to(device)
        save("vocoder", row, vocoder_infer(mel, vocoder, model_config, preprocess_config)[0])

    sweep = []
    if args.sweep_speakers and len(speakers) > 1:
        counts = Counter(r["speaker"] for r in read_filelist(os.path.join(folder, "train.txt")))
        sweep = [name for name, _ in counts.most_common(args.sweep_speakers)]
        print("Speakers in sweep:", sweep, flush=True)

    for name, path in checkpoints.items():
        model = get_model(argparse.Namespace(ckpt_file=path, restore_step=0), configs, device, train=False)
        for row in clips:
            targets = {kind: feature(folder, kind, row) for kind in ("pitch", "energy", "duration")}
            speaker = speakers[row["speaker"]]
            save("teacher:" + name, row, synthesize(model, vocoder, configs, row["phones"], speaker, targets))
            save("free:" + name, row, synthesize(model, vocoder, configs, row["phones"], speaker))
        for speaker in sweep:
            for row in clips[: args.sweep_clips]:
                wav = synthesize(model, vocoder, configs, row["phones"], speakers[speaker])
                save("sweep:{}@{}".format(name, speaker), row, wav, speaker)
        del model
        print("Generated", name, len(manifest), "clips so far", flush=True)
    return manifest


def transcribe(manifest, language, batch_size):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(WHISPER)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER).to(device)
    if device.type == "cuda":
        model = model.half()
    model.eval()
    for start in range(0, len(manifest), batch_size):
        batch = manifest[start:start + batch_size]
        audio = [librosa.load(item["path"], sr=16000)[0] for item in batch]
        features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        features = features.to(device).to(model.dtype)
        with torch.no_grad():
            tokens = model.generate(features, language=WHISPER_LANGUAGE[language], task="transcribe",
                                    max_new_tokens=200)
        for item, text in zip(batch, processor.batch_decode(tokens, skip_special_tokens=True)):
            reference, heard = normalise(item["text"]), normalise(text)
            item["heard"] = text.strip()
            item["wer"] = error_rate(reference.split(), heard.split())
            item["cer"] = error_rate(reference, heard)
        print("Transcribed", min(start + batch_size, len(manifest)), "of", len(manifest), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def predict_mos(manifest):
    """Naturalness is optional: a failed download must not lose the error rates."""
    try:
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
        predictor.eval()
        for item in manifest:
            wave = librosa.load(item["path"], sr=16000)[0]
            with torch.no_grad():
                score = predictor(torch.from_numpy(wave)[None].to(device), 16000)
            item["utmos"] = float(score.reshape(-1)[0])
    except Exception as error:
        print("UTMOS unavailable:", repr(error), flush=True)


def mean_interval(values):
    """Mean and 95% bootstrap interval."""
    values = np.array(values, dtype=float)
    picks = np.random.default_rng(1234).integers(0, len(values), (2000, len(values)))
    low, high = np.percentile(values[picks].mean(axis=1), [2.5, 97.5])
    return [round(float(values.mean()), 4), round(float(low), 4), round(float(high), 4)]


def summarise(manifest):
    groups = defaultdict(list)
    for item in manifest:
        groups[item["condition"]].append(item)
    summary = {}
    for condition, items in groups.items():
        summary[condition] = {"clips": len(items)}
        for metric in ("wer", "cer", "utmos"):
            values = [item[metric] for item in items if metric in item]
            if values:
                summary[condition][metric] = mean_interval(values)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=sorted(WHISPER_LANGUAGE))
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--audio_dir", required=True, help="scratch folder for every generated clip")
    parser.add_argument("--clips", type=int, default=60)
    parser.add_argument("--sweep_speakers", type=int, default=0)
    parser.add_argument("--sweep_clips", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--skip_scoring", action="store_true")
    args = parser.parse_args()

    configs = load_config(os.path.join(ROOT, args.language, "config.yaml"), args.work_dir)[:3]
    check_vocoder_compatible(configs[0])
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoints = dict(item.split("=", 1) for item in args.ckpt)
    manifest = generate(args, configs, checkpoints)
    if not args.skip_scoring:
        transcribe(manifest, args.language, args.batch_size)
        predict_mos(manifest)

    fields = ["condition", "id", "speaker", "wer", "cer", "utmos", "text", "heard"]
    with open(os.path.join(args.results_dir, "scores.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    summary = summarise(manifest)
    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    # A few clips per main condition, so the scores can be checked by ear
    for condition in summary:
        if condition.startswith("sweep"):
            continue
        items = [item for item in manifest if item["condition"] == condition][:LISTEN_CLIPS]
        target = os.path.join(args.results_dir, "listen", condition.replace(":", "_"))
        os.makedirs(target, exist_ok=True)
        for item in items:
            os.replace(item["path"], os.path.join(target, item["id"] + ".wav"))

    for condition, values in sorted(summary.items()):
        print(condition, json.dumps(values), flush=True)


if __name__ == "__main__":
    main()
