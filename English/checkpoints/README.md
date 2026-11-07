# English checkpoint

The checkpoint is too large for this repository. Download `english_step60000.pth.tar` from
https://huggingface.co/perceptron01/IIITH-FastSpeech2-English-Telugu and place it in this folder.

- Step 60,000: the end of the learning-rate schedule. Training stopped at step 54,153 at Kaggle's
  10-hour limit and was resumed with its optimizer state, with drops at 50,000, 55,000 and 57,500.
- The file includes the optimizer state, so training can be resumed from it.
- Default voice: `spk_f0bb6d4e` (see `../assets/target_speaker.txt`).

```bash
python English/inference.py --ckpt_file English/checkpoints/english_step60000.pth.tar --results_dir out/English
```
