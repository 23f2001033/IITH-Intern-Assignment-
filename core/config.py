import os

import yaml


def load_config(config_path, work_dir):
    """Read a language's config.yaml.

    Returns (preprocess_config, model_config, train_config, extra). The first
    three have the layout the reference code expects; paths are resolved
    against work_dir, so the same config runs locally and on Kaggle.
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    preprocess_config = config["preprocess"]
    train_config = config["train"]
    for section in (preprocess_config, train_config):
        for name, path in section["path"].items():
            section["path"][name] = os.path.join(work_dir, path)
    return preprocess_config, config["model"], train_config, config


def default_work_dir(language_dir):
    return os.environ.get("TTS_WORK_DIR", os.path.join(language_dir, "work"))
