"""Turn the training logs into training_loss.png and validation_loss.png."""
import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

NAMES = ["Total Loss", "Mel Loss", "Mel PostNet Loss", "Pitch Loss", "Energy Loss", "Duration Loss"]


def read_log(path):
    """Return {step: [six losses]}. A later line wins, so resumed runs stay consistent."""
    rows = {}
    with open(path) as f:
        for line in f:
            step = re.search(r"Step (\d+)", line)
            values = [re.search(r"{}: ([\d.]+)".format(n), line) for n in NAMES]
            if step and all(values):
                rows[int(step.group(1))] = [float(v.group(1)) for v in values]
    return dict(sorted(rows.items()))


def smooth(values, window):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot(rows, title, path, window):
    steps = list(rows)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for i, (ax, name) in enumerate(zip(axes.flat, NAMES)):
        values = [rows[s][i] for s in steps]
        if window > 1:
            ax.plot(steps, values, color="#2f6fb0", alpha=0.2, lw=0.7)
        ax.plot(steps, smooth(values, window), color="#2f6fb0", lw=1.4)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Training step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overlay(train, val, title, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for i, (ax, name) in enumerate(zip(axes.flat, NAMES)):
        ax.plot(list(train), smooth([v[i] for v in train.values()], 20), lw=1.2, label="train")
        ax.plot(list(val), [v[i] for v in val.values()], lw=1.2, label="validation")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].legend()
    for ax in axes[1]:
        ax.set_xlabel("Training step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, help="folder with train/log.txt and val/log.txt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", required=True, help="model name for the figure titles")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = read_log(os.path.join(args.log_dir, "train", "log.txt"))
    val = read_log(os.path.join(args.log_dir, "val", "log.txt"))
    plot(train, args.name + ": training loss", os.path.join(args.out_dir, "training_loss.png"), 20)
    plot(val, args.name + ": validation loss", os.path.join(args.out_dir, "validation_loss.png"), 1)
    plot_overlay(train, val, args.name + ": training vs validation loss",
                 os.path.join(args.out_dir, "train_vs_validation_loss.png"))


if __name__ == "__main__":
    main()
