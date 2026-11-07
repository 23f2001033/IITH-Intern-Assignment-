"""Write the Kaggle notebooks, so their cells stay under version control as plain text."""
import json
import os

REPO = "https://github.com/23f2001033/IITH-Intern-Assignment-.git"

SETUP = '''import glob, os, shutil, subprocess, time
REPO = "{repo}"
PROJECT = "/tmp/project"
shutil.rmtree(PROJECT, ignore_errors=True)
subprocess.run(["git", "clone", "--depth", "1", REPO, PROJECT], check=True)
print(subprocess.run(["git", "-C", PROJECT, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout)
subprocess.run("pip install -q pyworld tgt g2p-en inflect unidecode", shell=True, check=True)
'''.format(repo=REPO)

PREPARE = [
    '''# 1. QUICK_TRIAL = True tests the whole pipeline on a few minutes of audio.
#    Use False for the real run, then Save Version > Save & Run All.
QUICK_TRIAL = False
LANGUAGES = ["English", "Telugu"]
HOURS_SCALE = 0.05 if QUICK_TRIAL else 1.0
CORPORA = {"English": ("ai4bharat/Svarah", "svarah"), "Telugu": ("SPRINGLab/IndicTTS_Telugu", "telugu")}
''',
    SETUP,
    '''# 2. Install Montreal Forced Aligner
MFA = "/tmp/bin/micromamba run -r /tmp/mamba -n mfa mfa"
if not os.path.exists("/tmp/mamba/envs/mfa"):
    subprocess.run("curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp bin/micromamba", shell=True, check=True)
    subprocess.run("/tmp/bin/micromamba create -y -q -r /tmp/mamba -n mfa -c conda-forge montreal-forced-aligner", shell=True, check=True)
print(subprocess.run(MFA + " version", shell=True, capture_output=True, text=True).stdout)
''',
    '''# 3. Download both datasets (your HF_TOKEN secret must be attached to this notebook)
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
subprocess.run("pip install -q -U huggingface_hub", shell=True, check=True)
for repo, name in (CORPORA[language] for language in LANGUAGES):
    if not glob.glob("/tmp/raw/{}/data/*.parquet".format(name)):
        subprocess.run("hf download {} --repo-type dataset --local-dir /tmp/raw/{}".format(repo, name), shell=True, check=True)
del os.environ["HF_TOKEN"]
''',
    '''# 4. Prepare, align and extract features, with the languages running side by side.
#    Each language gets its own MFA folder because both corpora are called raw_data.
import re
os.makedirs("/kaggle/working/logs", exist_ok=True)
jobs = {}
for language in LANGUAGES:
    corpus = "/tmp/raw/" + CORPORA[language][1]
    work = "/tmp/tts/" + language
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree("/tmp/mfa_" + language, ignore_errors=True)
    command = [
        "python", "{}/{}/preprocess.py".format(PROJECT, language), "--stage", "all",
        "--corpus_path", corpus, "--work_dir", work, "--mfa_command", MFA,
        "--jobs", str(4 // len(LANGUAGES)), "--hours_scale", str(HOURS_SCALE),
    ]
    log = open("/kaggle/working/logs/prepare_{}.log".format(language), "w")
    env = dict(os.environ, MFA_ROOT_DIR="/tmp/mfa_" + language)
    jobs[language] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)

def latest_step(language):
    with open("/kaggle/working/logs/prepare_{}.log".format(language), "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4000))
        text = f.read().decode(errors="replace")
    lines = [line.strip() for line in text.replace("\\r", "\\n").split("\\n") if line.strip()]
    return lines[-1][:100] if lines else "(starting)"

while any(p.poll() is None for p in jobs.values()):
    time.sleep(120)
    for language, p in jobs.items():
        print(time.strftime("%H:%M"), language, "running" if p.poll() is None else p.returncode, "|", latest_step(language), flush=True)
print({k: p.returncode for k, p in jobs.items()})
for language in jobs:
    print("=" * 30, language)
    print(open("/kaggle/working/logs/prepare_{}.log".format(language), errors="replace").read()[-2500:])
''',
    '''# 5. Pack what training needs into one file per language
for language, job in jobs.items():
    if job.returncode != 0:
        print("SKIPPED", language, "- it failed, see its log above")
        continue
    work = "/tmp/tts/" + language
    subprocess.run(["tar", "-cf", "/kaggle/working/{}_preprocessed.tar".format(language), "-C", work, "preprocessed_data"], check=True)
    os.makedirs("/kaggle/working/reports/" + language, exist_ok=True)
    shutil.copy(work + "/preprocessed_data/filelists/data_report.json", "/kaggle/working/reports/" + language)
    shutil.copy(work + "/preprocessed_data/feature_summary.json", "/kaggle/working/reports/" + language)
    for name in ("train.ids", "val.ids"):
        shutil.copy(work + "/preprocessed_data/filelists/" + name, "/kaggle/working/reports/" + language)
    print(language, open(work + "/preprocessed_data/feature_summary.json").read())
subprocess.run("ls -la /kaggle/working; du -sh /kaggle/working/*", shell=True)
''',
]

TRAIN = [
    '''# Settings: use GPU T4 x2, and add the prepare notebook's output as an input.
# Each language in LANGUAGES trains on its own GPU.
LANGUAGES = ["English", "Telugu"]
MAX_HOURS = 10
RESTORE_STEP = {"English": 0, "Telugu": 0}
''',
    SETUP,
    '''# Unpack the features. Checkpoints and logs live in /kaggle/working so Kaggle keeps them.
subprocess.run("nvidia-smi", shell=True)
subprocess.run("find /kaggle/input -maxdepth 6 -not -name '*.npy' | head -40", shell=True)

def unpack(language, work):
    """Kaggle can keep a tar as it is or unpack it when an output becomes an input."""
    tars = glob.glob("/kaggle/input/**/{}_preprocessed.tar*".format(language), recursive=True)
    tars = [path for path in tars if os.path.isfile(path)]
    if tars:
        subprocess.run(["tar", "-xf", tars[0], "-C", work], check=True)
        return
    folders = glob.glob("/kaggle/input/**/preprocessed_data/feature_summary.json", recursive=True)
    folders = [os.path.dirname(path) for path in folders]
    folders = [path for path in folders if language in path] or folders
    assert len(folders) == 1, "Expected one {} input, found: {}".format(language, folders)
    shutil.copytree(folders[0], work + "/preprocessed_data", dirs_exist_ok=True)

for language in LANGUAGES:
    work = "/tmp/tts/" + language
    os.makedirs(work, exist_ok=True)
    unpack(language, work)
    for name in ("ckpt", "log"):
        keep = "/kaggle/working/{}/{}".format(language, name)
        os.makedirs(keep, exist_ok=True)
        if not os.path.exists(work + "/" + name):
            os.symlink(keep, work + "/" + name)
    print(language, open(work + "/preprocessed_data/feature_summary.json").read())
''',
    '''# Train both models at once, one per GPU
os.makedirs("/kaggle/working/logs", exist_ok=True)
jobs = {}
for gpu, language in enumerate(LANGUAGES):
    command = ["python", "{}/{}/train.py".format(PROJECT, language), "--work_dir", "/tmp/tts/" + language,
               "--max_hours", str(MAX_HOURS), "--restore_step", str(RESTORE_STEP[language])]
    log = open("/kaggle/working/logs/train_{}.log".format(language), "a")
    jobs[language] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))

def last_line(language):
    path = "/kaggle/working/{}/log/train/log.txt".format(language)
    return open(path).readlines()[-1].strip() if os.path.exists(path) else "(no steps logged yet)"

while any(p.poll() is None for p in jobs.values()):
    time.sleep(600)
    for language in jobs:
        print(time.strftime("%H:%M"), language, last_line(language), flush=True)
print({k: p.returncode for k, p in jobs.items()})
for language in jobs:
    print(open("/kaggle/working/logs/train_{}.log".format(language), errors="replace").read()[-1500:])
''',
    '''# Generate all deliverables from the best validation checkpoint
for language, job in jobs.items():
    if job.returncode != 0:
        print("SKIPPED", language)
        continue
    work = "/tmp/tts/" + language
    results = "/kaggle/working/Results/" + language
    ckpt = work + "/ckpt/best.pth.tar"
    print(language, "best step:", __import__("torch").load(ckpt, map_location="cpu", weights_only=False)["step"])
    subprocess.run(["python", "{}/{}/inference.py".format(PROJECT, language), "--work_dir", work,
                    "--ckpt_file", ckpt, "--results_dir", results], check=True)
    subprocess.run(["python", PROJECT + "/scripts/plot_losses.py", "--log_dir", work + "/log",
                    "--out_dir", results, "--name", language], check=True)
    assets = "/kaggle/working/{}/assets".format(language)
    os.makedirs(assets, exist_ok=True)
    for name in ("stats.json", "speakers.json", "target_speaker.txt"):
        shutil.copy(work + "/preprocessed_data/" + name, assets)

if "English" in jobs and jobs["English"].returncode == 0:
    subprocess.run(["python", PROJECT + "/English/inference.py", "--work_dir", "/tmp/tts/English",
                    "--ckpt_file", "/tmp/tts/English/ckpt/best.pth.tar", "--frontend", "stock",
                    "--results_dir", "/kaggle/working/Results/English/stock_frontend_comparison"], check=True)

subprocess.run("pip freeze > /kaggle/working/kaggle_pip_freeze.txt", shell=True)
subprocess.run("du -sh /kaggle/working/*; ls -la /kaggle/working/*/ckpt", shell=True)
''',
]


EVALUATE = [
    '''# Scores speech on validation sentences to show where quality is lost.
# Use GPU T4 x2. Add five inputs: prep-english, iiit-h, train-iith-eng, train-iith-tel, finish-english.
# QUICK_TEST = True runs a few clips in a few minutes to prove the setup first.
QUICK_TEST = False
LANGUAGES = ["English", "Telugu"]
CLIPS = 6 if QUICK_TEST else 60
SWEEP_SPEAKERS = 2 if QUICK_TEST else 12
SWEEP_CLIPS = 3 if QUICK_TEST else 30
# English voices to generate the test sentences with, from the last checkpoint
VOICES = ["spk_d270faea", "spk_8150bedf", "spk_f0bb6d4e", "spk_fae3eb5f"]
''',
    SETUP,
    '''# Unpack the features and find both checkpoints for each language
subprocess.run("nvidia-smi -L", shell=True)
subprocess.run("find /kaggle/input -maxdepth 6 -not -name '*.npy' | head -60", shell=True)

def checkpoints(language):
    """The newest numbered checkpoint of each training run, plus Telugu's best-validation one."""
    paths = glob.glob("/kaggle/input/**/{}/ckpt/*.pth.tar".format(language), recursive=True)
    newest = {}
    for path in paths:
        name = os.path.basename(path)
        if name[0].isdigit():
            item = (int(name.split(".")[0]), path)
            newest[os.path.dirname(path)] = max(newest.get(os.path.dirname(path), item), item)
    chosen = ["step{}={}".format(step, path) for step, path in sorted(newest.values())]
    if language == "Telugu":
        best = [path for path in paths if path.endswith("best.pth.tar")]
        assert len(best) == 1, best
        chosen = ["best=" + best[0]] + chosen
    assert chosen, "No {} checkpoints found".format(language)
    return chosen

for language in LANGUAGES:
    work = "/tmp/tts/" + language
    os.makedirs(work, exist_ok=True)
    tars = glob.glob("/kaggle/input/**/{}_preprocessed.tar*".format(language), recursive=True)
    tars = [path for path in tars if os.path.isfile(path)]
    assert len(tars) == 1, "Expected one {} features tar, found: {}".format(language, tars)
    subprocess.run(["tar", "-xf", tars[0], "-C", work], check=True)
    print(language, checkpoints(language))
''',
    '''# Generate the test sentences: English from its newest checkpoint in every candidate voice,
# Telugu from its best-validation checkpoint (step 13,000), which native listeners preferred
for language in LANGUAGES:
    last = checkpoints(language)[0 if language == "Telugu" else -1].split("=", 1)[1]
    voices = VOICES if language == "English" else [None]
    for voice in voices:
        command = ["python", "{}/{}/inference.py".format(PROJECT, language), "--work_dir", "/tmp/tts/" + language,
                   "--ckpt_file", last, "--results_dir",
                   "/kaggle/working/final/{}_{}".format(language, voice or os.path.basename(last).split(".")[0])]
        if voice:
            command += ["--speaker", voice]
        subprocess.run(command, check=True)
''',
    '''# Score both languages at once, one per GPU
os.makedirs("/kaggle/working/logs", exist_ok=True)
jobs = {}
for gpu, language in enumerate(LANGUAGES):
    command = ["python", PROJECT + "/scripts/evaluate_tts.py", "--language", language,
               "--work_dir", "/tmp/tts/" + language,
               "--results_dir", "/kaggle/working/evaluation/" + language,
               "--audio_dir", "/tmp/eval_audio/" + language, "--clips", str(CLIPS)]
    for item in checkpoints(language):
        command += ["--ckpt", item]
    if language == "English":
        command += ["--sweep_speakers", str(SWEEP_SPEAKERS), "--sweep_clips", str(SWEEP_CLIPS)]
    log = open("/kaggle/working/logs/evaluate_{}.log".format(language), "a")
    jobs[language] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))

def last_line(language):
    lines = open("/kaggle/working/logs/evaluate_{}.log".format(language), errors="replace").readlines()
    return lines[-1].strip()[:150] if lines else "(starting)"

while any(p.poll() is None for p in jobs.values()):
    time.sleep(60)
    for language in jobs:
        print(time.strftime("%H:%M"), language, last_line(language), flush=True)
print({k: p.returncode for k, p in jobs.items()})
for language in jobs:
    print(language, open("/kaggle/working/logs/evaluate_{}.log".format(language), errors="replace").read()[-6000:])
assert all(p.returncode == 0 for p in jobs.values()), "A scoring job failed; see the log above"
''',
]


FINISH = [
    '''# Finishes the English learning-rate schedule from its last checkpoint.
# The first run hit the 10-hour limit after only the first drop, at step 50,000.
# Use GPU T4 x2. Add two inputs: prep-english and train-iith-eng.
RESTORE_STEP = 54153
TOTAL_STEP = 60000
ANNEAL_STEPS = [50000, 55000, 57500]
''',
    SETUP,
    '''# Unpack the features and put the saved checkpoint where training expects it
subprocess.run("nvidia-smi -L", shell=True)
work = "/tmp/tts/English"
os.makedirs(work, exist_ok=True)
tars = [p for p in glob.glob("/kaggle/input/**/English_preprocessed.tar*", recursive=True) if os.path.isfile(p)]
saved = glob.glob("/kaggle/input/**/English/ckpt/{}.pth.tar".format(RESTORE_STEP), recursive=True)
assert len(tars) == 1 and len(saved) == 1, (tars, saved)
subprocess.run(["tar", "-xf", tars[0], "-C", work], check=True)
for name in ("ckpt", "log"):
    keep = "/kaggle/working/English/" + name
    os.makedirs(keep, exist_ok=True)
    os.symlink(keep, work + "/" + name)
shutil.copy(saved[0], work + "/ckpt/")
print(os.listdir(work + "/ckpt"))
''',
    '''# Train the remaining steps at the lower learning rates, about an hour
os.makedirs("/kaggle/working/logs", exist_ok=True)
command = ["python", PROJECT + "/English/train.py", "--work_dir", work, "--max_hours", "2",
           "--restore_step", str(RESTORE_STEP), "--total_step", str(TOTAL_STEP),
           "--anneal_steps"] + [str(step) for step in ANNEAL_STEPS]
log = open("/kaggle/working/logs/finish_English.log", "a")
job = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
while job.poll() is None:
    time.sleep(300)
    path = "/kaggle/working/English/log/train/log.txt"
    if os.path.exists(path):
        print(time.strftime("%H:%M"), open(path).readlines()[-1].strip(), flush=True)
print("return code", job.returncode)
print(open("/kaggle/working/logs/finish_English.log", errors="replace").read()[-1500:])
assert job.returncode == 0, "Training failed; see the log above"
''',
    '''# Generate the test sentences from the final checkpoint and from the best one in this stage
for name, ckpt in (("final", "{}.pth.tar".format(TOTAL_STEP)), ("best", "best.pth.tar")):
    subprocess.run(["python", PROJECT + "/English/inference.py", "--work_dir", work,
                    "--ckpt_file", work + "/ckpt/" + ckpt,
                    "--results_dir", "/kaggle/working/Results/English_" + name], check=True)
for path in glob.glob("/kaggle/working/English/ckpt/{}.pth.tar".format(RESTORE_STEP)):
    os.remove(path)
subprocess.run("du -sh /kaggle/working/*; ls -la /kaggle/working/English/ckpt", shell=True)
''',
]


def notebook(cells):
    return {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": cell.strip("\n").splitlines(True)}
            for cell in cells
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
    os.makedirs(folder, exist_ok=True)
    for name, cells in (("01_prepare_data.ipynb", PREPARE), ("02_train_and_generate.ipynb", TRAIN),
                         ("03_evaluate_speech.ipynb", EVALUATE), ("04_finish_english.ipynb", FINISH)):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            json.dump(notebook(cells), f, indent=1)
