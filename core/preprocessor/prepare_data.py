"""Select, clean and split a corpus, then write the files the aligner needs.

Steps: read the parquet shards, filter clips, resample everything to one rate,
split into train and validation with a fixed seed, verify there is no overlap,
and write wav/lab files, id lists, a pronunciation lexicon and data_report.json.
"""
import glob
import hashlib
import io
import json
import os
import random
import re
import shutil
from collections import defaultdict

import librosa
import numpy as np
import pyarrow.parquet as pq
import pyworld as pw
import soundfile as sf
from tqdm import tqdm

from text import english, telugu

_METADATA = [
    "gender", "age-group", "primary_language", "native_place_state",
    "native_place_district", "highest_qualification", "job_category",
    "occupation_domain",
]
_TELUGU_ALLOWED = re.compile(r"[ఀ-౿ ,.?!;:'\"]+")


def iter_rows(corpus_path):
    paths = sorted(glob.glob(os.path.join(corpus_path, "**", "*.parquet"), recursive=True))
    if not paths:
        raise FileNotFoundError("No parquet files under {}".format(corpus_path))
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            yield from batch.to_pylist()


def read_svarah_row(row, index, data):
    """Svarah has no speaker column. Speakers are recovered from the metadata
    fields, which give 115 distinct values for its 117 speakers."""
    stem = os.path.splitext(os.path.basename(row["audio_filepath"]["path"]))[0]
    combo = "|".join(str(row[k]) for k in _METADATA)
    return {
        "id": re.sub(r"[^A-Za-z0-9_]", "_", stem),
        "speaker": "spk_" + hashlib.md5(combo.encode("utf-8")).hexdigest()[:8],
        "recording": stem.split("_")[0],
        "gender": row["gender"],
        "text": row["text"],
        "audio": row["audio_filepath"]["bytes"],
    }


def read_telugu_row(row, index, data):
    if row["gender"] != data["gender_id"]:
        return None
    return {
        "id": "te_{:05d}".format(index),
        "speaker": data["speaker_name"],
        "recording": "te_{:05d}".format(index),
        "gender": data["speaker_name"],
        "text": row["text"].replace("﻿", "").replace("”", "").strip(),
        "audio": row["audio"]["bytes"],
    }


def english_text_ok(text):
    """Reject clips whose transcripts cannot be spoken reliably from the text,
    such as application IDs and dates of birth."""
    if re.search(r"\d{5,}|\d-\d", text):
        return False
    for token in re.findall(r"\w+", text):
        mixed = re.search(r"\d", token) and re.search(r"[a-zA-Z]", token)
        if mixed and not re.fullmatch(r"\d+(st|nd|rd|th)", token, re.IGNORECASE):
            return False
    return True


def telugu_text_ok(text):
    return _TELUGU_ALLOWED.fullmatch(text) is not None


LANGUAGES = {
    "en": {"reader": read_svarah_row, "text_ok": english_text_ok, "frontend": english},
    "te": {"reader": read_telugu_row, "text_ok": telugu_text_ok, "frontend": telugu},
}


def estimate_snr(wav, frame=1024, hop=512):
    """Rough SNR in dB: loud frames (speech) against quiet frames (background)."""
    n = 1 + max(0, len(wav) - frame) // hop
    frames = np.lib.stride_tricks.sliding_window_view(wav, frame)[::hop][:n]
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-8
    return float(20 * np.log10(np.percentile(rms, 95) / np.percentile(rms, 10)))


def stage_clips(corpus_path, stage_dir, language, data, sampling_rate):
    """Decode every usable clip once, resample it and record its statistics."""
    lang = LANGUAGES[language]
    rejected = defaultdict(int)
    records, seen, source_rates = [], set(), defaultdict(int)
    for index, row in enumerate(tqdm(iter_rows(corpus_path), desc="Reading clips")):
        rec = lang["reader"](row, index, data)
        if rec is None:
            rejected["other_speaker"] += 1
            continue
        if rec["id"] in seen:
            rejected["duplicate_id"] += 1
            continue
        seen.add(rec["id"])

        info = sf.info(io.BytesIO(rec["audio"]))
        duration = info.duration
        if not data["min_seconds"] <= duration <= data["max_seconds"]:
            rejected["duration_out_of_range"] += 1
            continue
        if not lang["text_ok"](rec["text"]):
            rejected["unsuitable_text"] += 1
            continue
        words = lang["frontend"].corpus_words(rec["text"])
        if not words:
            rejected["empty_text"] += 1
            continue

        wav, rate = sf.read(io.BytesIO(rec.pop("audio")), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        source_rates[rate] += 1
        if rate != sampling_rate:
            wav = librosa.resample(wav, orig_sr=rate, target_sr=sampling_rate)
        peak = np.max(np.abs(wav))
        if peak < 1e-3:
            rejected["silent"] += 1
            continue
        wav = wav / peak * 0.95

        rec.update(
            duration=len(wav) / sampling_rate,
            words=words,
            snr=estimate_snr(wav),
            chars_per_second=sum(len(w) for w in words) / duration,
            text_key=" ".join(words),
        )
        os.makedirs(os.path.join(stage_dir, rec["speaker"]), exist_ok=True)
        sf.write(
            os.path.join(stage_dir, rec["speaker"], rec["id"] + ".wav"),
            wav, sampling_rate, subtype="PCM_16",
        )
        records.append(rec)
    return records, dict(rejected), dict(source_rates)


def quality_filter(records, data, rejected):
    """Drop speaking-rate outliers (usually bad transcripts) and the noisiest clips."""
    rates = np.array([r["chars_per_second"] for r in records])
    low, high = np.percentile(rates, [1, 99])
    kept = [r for r in records if low <= r["chars_per_second"] <= high]
    rejected["speaking_rate_outlier"] = len(records) - len(kept)

    quantile = data.get("drop_noisiest_fraction", 0)
    if quantile:
        limit = np.quantile([r["snr"] for r in kept], quantile)
        before = len(kept)
        kept = [r for r in kept if r["snr"] >= limit]
        rejected["low_snr"] = before - len(kept)
    return kept


def group_related(records):
    """Union-find over clips sharing a recording or a transcript.

    A group goes wholly to one split, so neighbouring chunks of one recording
    and repeated prompts can never appear on both sides.
    """
    parent = list(range(len(records)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first = {}
    for i, rec in enumerate(records):
        for key in (("recording", rec["recording"]), ("text", rec["text_key"])):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i

    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(records[i])
    return sorted(groups.values(), key=lambda g: min(r["id"] for r in g))


def split_records(records, data):
    """Fixed-seed random split, filled by cumulative duration up to each target."""
    groups = group_related(records)
    random.Random(data["seed"]).shuffle(groups)

    train, val = [], []
    train_seconds = val_seconds = 0.0
    train_target = data["train_hours"] * 3600
    val_target = data["val_hours"] * 3600
    max_val_group = data.get("max_val_group_seconds", 600)
    for group in groups:
        seconds = sum(r["duration"] for r in group)
        if val_seconds < val_target and seconds <= max_val_group:
            val.extend(group)
            val_seconds += seconds
        elif train_seconds < train_target:
            train.extend(group)
            train_seconds += seconds

    # A large group can overshoot the target; dropping clips cannot create overlap
    random.Random(data["seed"]).shuffle(train)
    while train_seconds > train_target * 1.01:
        train_seconds -= train.pop()["duration"]

    # A speaker embedding is only trained on training clips, so every validation
    # speaker has to be present in training as well.
    train_speakers = {r["speaker"] for r in train}
    val = [r for r in val if r["speaker"] in train_speakers]
    return train, val


def verify_split(train, val, test_sentences, frontend, raw_dir, sampling_rate):
    """Fail loudly on any leakage; the returned counts go into data_report.json."""
    checks = {}
    for name, key in (("id", "id"), ("text", "text_key"), ("recording", "recording")):
        overlap = {r[key] for r in train} & {r[key] for r in val}
        checks["{}_overlap".format(name)] = len(overlap)
        assert not overlap, "{} overlap between train and val: {}".format(name, list(overlap)[:5])

    test_keys = {" ".join(frontend.corpus_words(t)) for t in test_sentences}
    leaked = test_keys & {r["text_key"] for r in train + val}
    checks["test_sentences_in_data"] = len(leaked)
    assert not leaked, "Test sentence found in the data: {}".format(leaked)

    missing = {r["speaker"] for r in val} - {r["speaker"] for r in train}
    checks["val_speakers_missing_from_train"] = len(missing)
    assert not missing

    rates = {
        sf.info(os.path.join(raw_dir, r["speaker"], r["id"] + ".wav")).samplerate
        for r in train + val
    }
    checks["wav_sample_rates"] = sorted(rates)
    assert rates == {sampling_rate}, rates
    return checks


def median_f0(records, raw_dir, seed, n=40):
    """Median F0 of a sample of clips, as a check on the speaker's gender label."""
    sample = random.Random(seed).sample(records, min(n, len(records)))
    values = []
    for rec in sample:
        wav, rate = sf.read(os.path.join(raw_dir, rec["speaker"], rec["id"] + ".wav"))
        f0, _ = pw.dio(wav.astype(np.float64), rate)
        values.extend(f0[f0 > 0])
    return float(np.median(values)) if values else None


def write_lexicon(records, language, lexicon_path, source_lexicon, test_sentences):
    words = {w for r in records for w in r["words"]}
    frontend = LANGUAGES[language]["frontend"]
    for sentence in test_sentences:
        words.update(frontend.corpus_words(sentence))

    source = english.read_lexicon(source_lexicon) if language == "en" else None
    unknown = 0
    with open(lexicon_path, "w", encoding="utf-8") as f:
        for word in tqdm(sorted(words), desc="Writing lexicon"):
            if language == "en":
                unknown += word not in source
                prons = english.pronunciations(word, source)
            else:
                prons = [telugu.word_to_phones(word)]
            for pron in prons:
                if pron:
                    f.write("{}\t{}\n".format(word, " ".join(pron)))
    return {"words": len(words), "words_from_g2p": unknown}


def summarise(records):
    durations = [r["duration"] for r in records]
    by_speaker = defaultdict(float)
    for r in records:
        by_speaker[r["speaker"]] += r["duration"]
    return {
        "clips": len(records),
        "hours": round(sum(durations) / 3600, 3),
        "speakers": len(by_speaker),
        "min_seconds": round(min(durations), 2),
        "median_seconds": round(float(np.median(durations)), 2),
        "max_seconds": round(max(durations), 2),
        "median_snr_db": round(float(np.median([r["snr"] for r in records])), 1),
        "gender_hours": {
            g: round(sum(r["duration"] for r in records if r["gender"] == g) / 3600, 3)
            for g in sorted({r["gender"] for r in records})
        },
    }


def prepare(preprocess_config, config, corpus_path, test_sentences, source_lexicon):
    data = config["data"]
    language = preprocess_config["preprocessing"]["text"]["language"]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    raw_dir = preprocess_config["path"]["raw_path"]
    filelist_dir = preprocess_config["path"]["filelist_path"]
    stage_dir = raw_dir + "_staging"

    for path in (raw_dir, stage_dir):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(filelist_dir, exist_ok=True)

    records, rejected, source_rates = stage_clips(
        corpus_path, stage_dir, language, data, sampling_rate
    )
    usable = len(records)
    usable_hours = sum(r["duration"] for r in records) / 3600
    records = quality_filter(records, data, rejected)
    train, val = split_records(records, data)

    for rec in train + val:
        speaker_dir = os.path.join(raw_dir, rec["speaker"])
        os.makedirs(speaker_dir, exist_ok=True)
        os.replace(
            os.path.join(stage_dir, rec["speaker"], rec["id"] + ".wav"),
            os.path.join(speaker_dir, rec["id"] + ".wav"),
        )
        with open(os.path.join(speaker_dir, rec["id"] + ".lab"), "w", encoding="utf-8") as f:
            f.write(" ".join(rec["words"]))
    shutil.rmtree(stage_dir, ignore_errors=True)

    frontend = LANGUAGES[language]["frontend"]
    checks = verify_split(train, val, test_sentences, frontend, raw_dir, sampling_rate)

    for name, rows in (("train", train), ("val", val)):
        with open(os.path.join(filelist_dir, name + ".ids"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(r["id"] for r in rows)) + "\n")

    # The voice used for the test sentences: the speaker with the most training audio
    hours = defaultdict(float)
    for rec in train:
        hours[rec["speaker"]] += rec["duration"] / 3600
    target_speaker = max(sorted(hours), key=hours.get)
    target_path = os.path.join(preprocess_config["path"]["preprocessed_path"], "target_speaker.txt")
    with open(target_path, "w") as f:
        f.write(target_speaker)

    lexicon = write_lexicon(
        train + val, language, preprocess_config["path"]["lexicon_path"],
        source_lexicon, test_sentences,
    )
    report = {
        "dataset": preprocess_config["dataset"],
        "selection_settings": data,
        "sampling_rate": sampling_rate,
        "source_sample_rates": {str(k): v for k, v in source_rates.items()},
        "usable_before_quality_filter": {"clips": usable, "hours": round(usable_hours, 3)},
        "rejected": rejected,
        "train": summarise(train),
        "val": summarise(val),
        "checks": checks,
        "target_speaker": target_speaker,
        "target_speaker_train_hours": round(hours[target_speaker], 3),
        "median_f0_hz": median_f0(train, raw_dir, data["seed"]),
        "lexicon": lexicon,
    }
    with open(os.path.join(filelist_dir, "data_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    for name, target in (("train", data["train_hours"]), ("val", data["val_hours"])):
        if report[name]["hours"] < 0.9 * target:
            print("WARNING: {} has {} h, target {} h".format(name, report[name]["hours"], target))
    return report
