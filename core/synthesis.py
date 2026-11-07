"""Generate the test sentences and plot the variance adaptor's predictions."""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import wavfile

from config import default_work_dir, load_config
from preprocess_cli import read_test_sentences
from text import english, telugu, text_to_sequence
from utils.model import check_vocoder_compatible, get_model, get_vocoder, vocoder_infer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = os.path.dirname(os.path.abspath(__file__))


def text_to_groups(text, language, frontend):
    """Return [(word, phones)] for a sentence."""
    if language == "te":
        return telugu.text_to_phone_groups(text)
    if frontend == "stock":
        lexicon = english.read_lexicon(os.path.join(ROOT, "lexicon", "librispeech-lexicon.txt"))
        return [("", english.stock_phones(text, lexicon))]
    return english.text_to_phone_groups(text)


def speaker_id(preprocess_config, name):
    folder = preprocess_config["path"]["preprocessed_path"]
    with open(os.path.join(folder, "speakers.json"), encoding="utf-8") as f:
        speakers = json.load(f)
    if name is None:
        target = os.path.join(folder, "target_speaker.txt")
        if os.path.exists(target):
            with open(target) as f:
                name = f.read().strip()
        else:
            name = sorted(speakers)[0]
    return speakers[name], name


PAD_SECONDS = 0.3
FADE_SECONDS = 0.005


def add_silence(wav, rate, seconds=PAD_SECONDS):
    """Fade the edges and add silence before and after the speech.

    Training clips had their leading and trailing silence trimmed, so the model starts
    speaking on the very first sample. Many players and earphones clip the first
    fraction of a second, which listeners heard as a missing first word.
    """
    wav = wav.astype(np.float32)
    fade = min(int(rate * FADE_SECONDS), len(wav) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wav[:fade] *= ramp
        wav[-fade:] *= ramp[::-1]
    pad = np.zeros(int(rate * seconds), dtype=np.float32)
    return np.concatenate([pad, wav, pad]).astype(np.int16)


def synthesize_one(model, vocoder, configs, sentence_id, text, groups, speaker):
    preprocess_config, model_config, _ = configs
    phones = [p for _, group in groups for p in group]
    sequence = np.array(text_to_sequence("{" + " ".join(phones) + "}", []))

    batch = (
        torch.from_numpy(np.array([speaker])).long().to(device),
        torch.from_numpy(sequence[None, :]).long().to(device),
        torch.from_numpy(np.array([len(sequence)])).to(device),
        len(sequence),
    )
    with torch.no_grad():
        output = model(*batch)

    mel_lens = output[9]
    hop = preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav = vocoder_infer(
        output[1].transpose(1, 2), vocoder, model_config, preprocess_config,
        lengths=mel_lens * hop,
    )[0]

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
    pitch_mean, pitch_std = stats["pitch"][2:4]
    energy_mean, energy_std = stats["energy"][2:4]

    rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    frames = output[5][0].detach().cpu().numpy()
    speech_seconds = round(len(wav) / rate, 2)
    wav = add_silence(wav, rate)
    return wav, {
        "id": sentence_id,
        "text": text,
        "phones": phones,
        "groups": [[word, len(group)] for word, group in groups],
        "duration_frames": frames.astype(int).tolist(),
        "duration_ms": (frames * hop / rate * 1000).round(1).tolist(),
        # The model predicts normalised values; convert back to physical units
        "f0_hz": (output[2][0].detach().cpu().numpy() * pitch_std + pitch_mean).round(1).tolist(),
        "energy": (output[3][0].detach().cpu().numpy() * energy_std + energy_mean).round(3).tolist(),
        "total_seconds": speech_seconds,
        "silence_padding_seconds": PAD_SECONDS,
    }


def display_word(word, phones):
    """Matplotlib's default font has no Telugu glyphs, so label those words by their phones."""
    if word in ("ayyem", "peeyem"):
        return "AM" if word == "ayyem" else "PM"
    return word if word.isascii() else "".join(phones)


def word_spans(item):
    spans, start = [], 0
    for word, count in item["groups"]:
        spans.append((display_word(word, item["phones"][start:start + count]), start, start + count))
        start += count
    return spans


def plot_item(item, kind, path):
    phones = item["phones"]
    n = len(phones)
    fig, ax = plt.subplots(figsize=(max(10, 0.26 * n), 4.2))
    x = np.arange(n)
    silent = [p == "sp" for p in phones]
    colors = ["#bbbbbb" if s else "#2f6fb0" for s in silent]

    if kind == "duration":
        ax.bar(x, item["duration_ms"], color=colors, width=0.8)
        ax.set_ylabel("Predicted duration (ms)")
    elif kind == "pitch":
        ax.step(x, item["f0_hz"], where="mid", color="#c8452f")
        ax.scatter(x, item["f0_hz"], c=["#bbbbbb" if s else "#c8452f" for s in silent], s=14, zorder=3)
        ax.set_ylabel("Predicted F0 (Hz)")
    else:
        ax.step(x, item["energy"], where="mid", color="#6a3fa0")
        ax.scatter(x, item["energy"], c=["#bbbbbb" if s else "#6a3fa0" for s in silent], s=14, zorder=3)
        ax.set_ylabel("Predicted energy (frame L2 norm)")

    ax.set_xticks(x)
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ax.set_xlim(-0.7, n - 0.3)
    ax.grid(axis="y", alpha=0.3)
    top = ax.get_ylim()[1]
    for word, start, end in word_spans(item):
        if start:
            ax.axvline(start - 0.5, color="#888888", lw=0.6, ls=":")
        if word:
            ax.text((start + end - 1) / 2, top, word, ha="center", va="bottom", fontsize=7, rotation=30)
    ax.set_title(
        "{}: predicted {} per phoneme ({:.2f} s)".format(item["id"], kind, item["total_seconds"]),
        pad=38, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_word_comparison(items, word, label, path):
    """Side-by-side variance predictions for the same written word in two contexts."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 7))
    keys = [("duration_ms", "Duration (ms)"), ("f0_hz", "F0 (Hz)"), ("energy", "Energy")]
    for offset, item in zip((-0.2, 0.2), items):
        start = 0
        for w, count in item["groups"]:
            if w == word:
                break
            start += count
        else:
            raise ValueError("{} not found in {}".format(word, item["id"]))
        phones = item["phones"][start:start + count]
        x = np.arange(len(phones))
        for ax, (key, axis_label) in zip(axes, keys):
            ax.bar(x + offset, item[key][start:start + count], width=0.38, label=item["id"])
            ax.set_ylabel(axis_label)
            ax.set_xticks(x)
            ax.set_xticklabels(phones, fontsize=8)
    axes[0].legend()
    axes[0].set_title("Same word, two contexts: {}".format(label))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(language_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default=default_work_dir(language_dir))
    parser.add_argument("--ckpt_file", default=None, help="checkpoint file; overrides --restore_step")
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--assets_dir", default=os.path.join(language_dir, "assets"),
                        help="folder with stats.json and speakers.json, used when no preprocessed data is present")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--frontend", choices=["improved", "stock"], default="improved")
    parser.add_argument("--text", default=None, help="synthesise this text instead of the test sentences")
    args = parser.parse_args()

    preprocess_config, model_config, train_config, config = load_config(
        os.path.join(language_dir, "config.yaml"), args.work_dir
    )
    preprocessed = preprocess_config["path"]["preprocessed_path"]
    if not os.path.exists(os.path.join(preprocessed, "stats.json")):
        preprocess_config["path"]["preprocessed_path"] = args.assets_dir
    check_vocoder_compatible(preprocess_config)

    configs = (preprocess_config, model_config, train_config)
    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    language = preprocess_config["preprocessing"]["text"]["language"]
    speaker, speaker_name = speaker_id(preprocess_config, args.speaker)
    print("Speaker:", speaker_name)

    audio_dir = os.path.join(args.results_dir, "test_generated_audios")
    plot_dir = os.path.join(args.results_dir, "variance_adaptor_plots")
    for name in ("duration", "pitch", "energy", "values"):
        os.makedirs(os.path.join(plot_dir, name), exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    sentences = [("custom", args.text)] if args.text else read_test_sentences(language_dir)
    rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    items = {}
    for sentence_id, text in sentences:
        groups = text_to_groups(text, language, args.frontend)
        wav, item = synthesize_one(model, vocoder, configs, sentence_id, text, groups, speaker)
        items[sentence_id] = item
        wavfile.write(os.path.join(audio_dir, sentence_id + ".wav"), rate, wav)
        for kind in ("duration", "pitch", "energy"):
            plot_item(item, kind, os.path.join(plot_dir, kind, sentence_id + ".png"))
        with open(os.path.join(plot_dir, "values", sentence_id + ".json"), "w", encoding="utf-8") as f:
            json.dump(item, f, indent=1, ensure_ascii=False)
        print(sentence_id, item["total_seconds"], "s:", " ".join(item["phones"]))

    comparison = config.get("inference", {}).get("compare")
    if comparison and args.frontend == "improved" and not args.text:
        plot_word_comparison(
            [items[s] for s in comparison["sentences"]],
            comparison["word"],
            comparison["label"],
            os.path.join(plot_dir, "homograph_comparison.png"),
        )
