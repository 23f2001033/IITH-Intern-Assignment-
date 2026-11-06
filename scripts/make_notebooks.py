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
    for name, cells in (("01_prepare_data.ipynb", PREPARE), ("02_train_and_generate.ipynb", TRAIN)):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            json.dump(notebook(cells), f, indent=1)
