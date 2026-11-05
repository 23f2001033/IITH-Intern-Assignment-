import argparse
import glob
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import default_work_dir, load_config
from utils.model import check_vocoder_compatible, get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, optimizer, path):
    torch.save(
        {
            "model": model.module.state_dict(),
            "optimizer": optimizer._optimizer.state_dict(),
        },
        path,
    )


def prune_checkpoints(ckpt_path, keep):
    steps = sorted(
        int(os.path.basename(p).split(".")[0])
        for p in glob.glob(os.path.join(ckpt_path, "[0-9]*.pth.tar"))
    )
    for step in steps[:-keep]:
        os.remove(os.path.join(ckpt_path, "{}.pth.tar".format(step)))


def run(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs
    check_vocoder_compatible(preprocess_config)

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=train_config["optimizer"].get("num_workers", 0),
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    keep_checkpoints = train_config["step"].get("keep_checkpoints", 3)
    ckpt_path = train_config["path"]["ckpt_path"]

    # Kaggle kills a session at 12 hours, so stop cleanly before the limit
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    best_val_loss = float("inf")

    outer_bar = tqdm(total=total_step, desc="Training", position=0, mininterval=30)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                try:
                    # Forward
                    output = model(*(batch[2:]))

                    # Cal Loss
                    losses = Loss(batch, output)
                    total_loss = losses[0]

                    # Backward
                    total_loss = total_loss / grad_acc_step
                    total_loss.backward()
                except torch.cuda.OutOfMemoryError:
                    # One batch of very long clips should not end a 10-hour run
                    print("Step {}: out of memory, skipping this batch".format(step))
                    output = losses = total_loss = None
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue

                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message, val_losses = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    if val_losses[0] < best_val_loss:
                        best_val_loss = val_losses[0]
                        torch.save(
                            {"model": model.module.state_dict(), "step": step},
                            os.path.join(ckpt_path, "best.pth.tar"),
                        )

                    model.train()

                out_of_time = deadline is not None and time.time() > deadline
                if step % save_step == 0 or step == total_step or out_of_time:
                    save_checkpoint(
                        model, optimizer, os.path.join(ckpt_path, "{}.pth.tar".format(step))
                    )
                    prune_checkpoints(ckpt_path, keep_checkpoints)

                if step == total_step or out_of_time:
                    print("Stopped at step {}".format(step))
                    return
                step += 1
                outer_bar.update(1)

        epoch += 1


def main(language_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--work_dir", default=default_work_dir(language_dir))
    parser.add_argument("--max_hours", type=float, default=None)
    parser.add_argument("--total_step", type=int, default=None)
    args = parser.parse_args()

    preprocess_config, model_config, train_config, _ = load_config(
        os.path.join(language_dir, "config.yaml"), args.work_dir
    )
    if args.total_step:
        train_config["step"]["total_step"] = args.total_step
    run(args, (preprocess_config, model_config, train_config))
