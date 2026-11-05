import argparse
import os

from config import default_work_dir, load_config

ROOT = os.path.dirname(os.path.abspath(__file__))


def read_test_sentences(language_dir):
    with open(os.path.join(language_dir, "test_sentences.txt"), encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("|", 1) for line in f if line.strip()]
    return [(sentence_id, text) for sentence_id, text in rows]


def main(language_dir):
    parser = argparse.ArgumentParser(
        description="prepare: select and split clips; align: MFA; features: mel, pitch, energy"
    )
    parser.add_argument("--stage", choices=["prepare", "align", "features", "all"], default="all")
    parser.add_argument("--corpus_path", help="folder holding the dataset's parquet files")
    parser.add_argument("--work_dir", default=default_work_dir(language_dir))
    parser.add_argument("--mfa_command", default=os.environ.get("MFA_COMMAND", "mfa"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--hours_scale", type=float, default=1.0,
                        help="scale the target hours, for a quick trial of the whole pipeline")
    args = parser.parse_args()

    config_path = os.path.join(language_dir, "config.yaml")
    preprocess_config, _, _, config = load_config(config_path, args.work_dir)

    config["data"]["train_hours"] *= args.hours_scale
    config["data"]["val_hours"] *= args.hours_scale

    if args.stage in ("prepare", "all"):
        from preprocessor.prepare_data import prepare

        if not args.corpus_path:
            parser.error("--corpus_path is required for the prepare stage")
        prepare(
            preprocess_config,
            config,
            args.corpus_path,
            [text for _, text in read_test_sentences(language_dir)],
            os.path.join(ROOT, "lexicon", "librispeech-lexicon.txt"),
        )
    if args.stage in ("align", "all"):
        from preprocessor.align import align

        align(preprocess_config, config, args.mfa_command, args.jobs)
    if args.stage in ("features", "all"):
        from preprocessor.preprocessor import Preprocessor

        Preprocessor(preprocess_config).build_from_path()
