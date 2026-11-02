# FastSpeech 2 TTS for English (Svarah) and Telugu (IndicTTS)

Two FastSpeech 2 models, trained from scratch on about 5 h of training audio and 1 h of validation audio each. The code starts from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (fork with all changes: [sudarshan032/FastSpeech2](https://github.com/sudarshan032/FastSpeech2)).
The full write-up is in [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf).

| | English | Telugu |
|---|---|---|
| Data | Svarah, 115 speakers, 4.86 h train / 1.02 h val | IndicTTS Telugu male, 5.00 h train / 1.00 h val |
| Text → phonemes | CMUdict dictionary; MFA G2P for unknown words; part-of-speech rules for homographs | rule-based Telugu G2P (`fastspeech2/text/telugu.py`) |
| Alignment | MFA 3, pretrained English model | MFA 3, model trained on this data |
| Checkpoint | `English/checkpoints/41650.pth.tar` | `Telugu/checkpoints/37261.pth.tar` |

## Layout

```
English/, Telugu/     preprocess.py, train.py, inference.py, config.yaml, checkpoints/, logs/
Results/<lang>/       training_loss.png, validation_loss.png, test_generated_audios/,
                      variance_adaptor_plots/{duration,pitch,energy}/, variance_predictions.json
fastspeech2/          model code (fork of ming024/FastSpeech2 with the changes listed in the report)
data_prep/            the Kaggle scripts that built the datasets, alignments, features, training and results
Report/               Technical_Report.pdf
```

## Setup

```bash
uv sync                      # or: pip install -r requirements.txt
source .venv/bin/activate
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
cd fastspeech2/hifigan && unzip generator_universal.pth.tar.zip && cd ../..
```

The checkpoints (`English/checkpoints/41650.pth.tar`, `Telugu/checkpoints/37261.pth.tar`, 134 MB each, model weights only) are stored with Git LFS. After cloning, run `git lfs pull` to download them.

## Inference

The commands below synthesize the test sentences and write the audio, the variance-adaptor plots and the loss curves to `Results/<lang>/`.

```bash
python English/inference.py --restore_step 41650
python Telugu/inference.py  --restore_step 37261
```

To synthesize one custom sentence instead:

```bash
python English/inference.py --restore_step 41650 --text "Hello there." --speaker spk029 --out /tmp/en
python Telugu/inference.py  --restore_step 37261 --text "నమస్కారం, ఎలా ఉన్నారు?" --out /tmp/te
```

## Preprocessing and training

Raw audio and alignments are not stored in the repo. They are rebuilt by the `data_prep/` scripts, which ran as Kaggle kernels in this order:

1. `svarah_1_select.py`: select Svarah clips and make the train/val split.
2. `svarah_2_align.py`: MFA alignment.
3. `svarah_3_filter_preprocess.py`: filter clips and extract features.
4. `telugu_select_align_preprocess.py`: everything for Telugu.
5. `kaggle_train.py`, `kaggle_results.py`: training and results (`fastspeech2/make_results.py`).
6. `speaker_eval.py`: English per-speaker evaluation.

Once `fastspeech2/raw_data/<dataset>` and the TextGrids are in place:

```bash
python English/preprocess.py && python English/train.py --max_hours 6
python Telugu/preprocess.py  && python Telugu/train.py  --max_hours 6
```

All paths in `config.yaml` are relative to `fastspeech2/`. The submitted checkpoints are model weights only, for inference; resuming training with `--restore_step` needs a full checkpoint written by `train.py`.
