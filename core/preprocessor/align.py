"""Run Montreal Forced Aligner to obtain phoneme durations as TextGrid files."""
import glob
import os
import shlex
import subprocess


def _run(command):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def align(preprocess_config, config, mfa_command, jobs):
    settings = config["align"]
    corpus = preprocess_config["path"]["raw_path"]
    lexicon = preprocess_config["path"]["lexicon_path"]
    output = os.path.join(preprocess_config["path"]["preprocessed_path"], "TextGrid")
    os.makedirs(output, exist_ok=True)

    mfa = shlex.split(mfa_command)
    options = ["-j", str(jobs), "--beam", str(settings["beam"]),
               "--retry_beam", str(settings["retry_beam"]), "--overwrite"]
    if settings.get("single_speaker"):
        options.append("--single_speaker")

    if settings.get("acoustic_model"):
        # English: a pretrained acoustic model exists
        acoustic_model = settings["acoustic_model"]
        _run(mfa + ["model", "download", "acoustic", acoustic_model])
    else:
        # Telugu: no pretrained model exists, so train one on this corpus and lexicon
        acoustic_model = os.path.join(
            preprocess_config["path"]["preprocessed_path"], "acoustic_model.zip"
        )
        _run(mfa + ["train", corpus, lexicon, acoustic_model] + options)
    _run(mfa + ["align", corpus, lexicon, acoustic_model, output] + options)

    wavs = glob.glob(os.path.join(corpus, "*", "*.wav"))
    grids = glob.glob(os.path.join(output, "*", "*.TextGrid"))
    coverage = len(grids) / max(1, len(wavs))
    print("Aligned {} of {} clips ({:.1%})".format(len(grids), len(wavs), coverage))
    if coverage < 0.9:
        raise RuntimeError("Too few clips aligned; check the MFA log above")
