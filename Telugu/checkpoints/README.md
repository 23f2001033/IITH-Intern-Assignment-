# Telugu checkpoint

The checkpoint is too large for this repository. Download `telugu_step13000.pth.tar` from
https://huggingface.co/perceptron01/IIITH-FastSpeech2-English-Telugu and place it in this folder.

- Step 13,000: the lowest validation loss of the 32,000-step run, preferred by the native reviewer who
  could tell it apart from step 32,000 (Section 7 of the report).

```bash
python Telugu/inference.py --ckpt_file Telugu/checkpoints/telugu_step13000.pth.tar --results_dir out/Telugu
```
