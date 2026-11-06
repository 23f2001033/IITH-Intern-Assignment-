"""Run Montreal Forced Aligner to obtain phoneme durations as TextGrid files."""
import glob
import os
import shlex
import subprocess
import time

QUIET_LIMIT_SECONDS = 45 * 60
POLL_SECONDS = 60


def _newest_change(folder):
    newest = 0
    for path, _, names in os.walk(folder):
        for name in names:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(path, name)))
            except OSError:
                pass
    return newest


def _run(command):
    """Run MFA, stopping it if its working folder stops changing.

    When an MFA export worker crashes, the main process waits for it forever
    instead of exiting, so a return code alone cannot be relied on.
    """
    print("+", " ".join(command), flush=True)
    folder = os.environ.get("MFA_ROOT_DIR", os.path.expanduser("~/Documents/MFA"))
    process = subprocess.Popen(command)
    newest, changed_at = 0, time.time()
    while process.poll() is None:
        time.sleep(POLL_SECONDS)
        current = _newest_change(folder)
        if current > newest:
            newest, changed_at = current, time.time()
        elif time.time() - changed_at > QUIET_LIMIT_SECONDS:
            process.kill()
            raise RuntimeError("MFA stopped making progress; see the log above")
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def align(preprocess_config, config, mfa_command, jobs):
    settings = config["align"]
    corpus = preprocess_config["path"]["raw_path"]
    lexicon = preprocess_config["path"]["lexicon_path"]
    output = os.path.join(preprocess_config["path"]["preprocessed_path"], "TextGrid")
    os.makedirs(output, exist_ok=True)

    mfa = shlex.split(mfa_command)
    # --clean matters for Telugu: train and align share a database, and without it
    # each clip keeps both alignments, which overlap and crash the TextGrid export.
    options = ["-j", str(jobs), "--beam", str(settings["beam"]),
               "--retry_beam", str(settings["retry_beam"]), "--overwrite", "--clean"]
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
