"""Synthesize the assignment test sentences, plot the variance-adaptor predictions and the loss curves.

python3 make_results.py --lang en --restore_step 150000 --speakers spk073 spk012 --out ../Results/English
python3 make_results.py --lang te --restore_step 150000 --out ../Results/Telugu
"""
import argparse
import json
import os
import re
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
import yaml
from scipy.io import wavfile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from synthesize import preprocess_english, preprocess_telugu  # noqa: E402
from text import _id_to_symbol  # noqa: E402
from utils.model import get_model, get_vocoder, vocoder_infer  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# test sentences exactly as given in the assignment
TESTS = {
    "en": {
        "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
        "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
        "E3": "The rhyth m of the orchestra changed unexpectedly.",
        "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
    },
    "te": {
        "T1": "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "T2b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చ ింది.",
        "T3": "పులి అడ విలో నిశ్శ బ్దంగా నడిచింది.",
        "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
    },
}
DATASET = {"en": "Svarah", "te": "IndicTTS_Telugu"}


def load_configs(name):
    return tuple(
        yaml.load(open(f"config/{name}/{k}.yaml"), Loader=yaml.FullLoader)
        for k in ("preprocess", "model", "train")
    )


def bar_plot(path, phones, values, ylabel, title):
    fig, ax = plt.subplots(figsize=(max(8, len(phones) * 0.28), 3.2))
    ax.bar(range(len(phones)), values, color="#4C72B0")
    ax.set_xticks(range(len(phones)))
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(phones) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_log(path):
    steps, rows = [], []
    for line in open(path):
        m = re.search(r"Step (\d+)", line)
        vals = re.findall(r"Loss: ([\d.]+)", line)
        if m and len(vals) == 6:
            steps.append(int(m.group(1)))
            rows.append([float(v) for v in vals])
    return np.array(steps), np.array(rows)


def loss_plot(log_path, out_path, title):
    names = ["Total", "Mel", "Mel PostNet", "Pitch", "Energy", "Duration"]
    if not os.path.exists(log_path):
        return
    steps, rows = parse_log(log_path)
    if len(steps) == 0:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(steps, rows[:, i], lw=0.8)
        ax.set_title(names[i] + " loss", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(a, configs=None):
    name = DATASET[a.lang]
    configs = configs or load_configs(name)
    pp, mc, tc = configs
    hop, sr = pp["preprocessing"]["stft"]["hop_length"], pp["preprocessing"]["audio"]["sampling_rate"]
    stats = json.load(open(os.path.join(pp["path"]["preprocessed_path"], "stats.json")))
    spk_map = json.load(open(os.path.join(pp["path"]["preprocessed_path"], "speakers.json")))
    speakers = a.speakers or list(spk_map)[:1]

    model = get_model(SimpleNamespace(restore_step=a.restore_step), configs, device, train=False)
    vocoder = get_vocoder(mc, device)
    front = preprocess_english if a.lang == "en" else preprocess_telugu

    for sub in ("test_generated_audios", "variance_adaptor_plots/duration",
                "variance_adaptor_plots/pitch", "variance_adaptor_plots/energy"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    details = {}
    for sid, sentence in TESTS[a.lang].items():
        seq = front(sentence, pp)
        phones = [_id_to_symbol[i].lstrip("@") for i in seq]
        for k, spk in enumerate(speakers):
            texts = torch.from_numpy(np.array([seq])).long().to(device)
            src_lens = torch.tensor([len(seq)]).to(device)
            spk_t = torch.tensor([spk_map[spk]]).long().to(device)
            with torch.no_grad():
                out = model(spk_t, texts, src_lens, len(seq))
            wav = vocoder_infer(out[1].transpose(1, 2), vocoder, mc, pp, lengths=out[9] * hop)[0]
            fname = sid if k == 0 else f"{sid}_{spk}"
            wavfile.write(os.path.join(a.out, "test_generated_audios", fname + ".wav"), sr, wav)
            if k:
                continue
            # predictions are per phoneme; undo the z-score normalization from preprocessing
            dur_ms = out[5][0].cpu().numpy() * hop / sr * 1000
            pitch_hz = out[2][0].cpu().numpy() * stats["pitch"][3] + stats["pitch"][2]
            energy = out[3][0].cpu().numpy() * stats["energy"][3] + stats["energy"][2]
            tag = f"{sid} ({spk})" if len(spk_map) > 1 else sid
            base = os.path.join(a.out, "variance_adaptor_plots")
            bar_plot(f"{base}/duration/{sid}.png", phones, dur_ms, "duration (ms)", f"{tag}: predicted duration per phoneme")
            bar_plot(f"{base}/pitch/{sid}.png", phones, pitch_hz, "F0 (Hz)", f"{tag}: predicted pitch (F0) per phoneme")
            bar_plot(f"{base}/energy/{sid}.png", phones, energy, "energy (mel-frame L2 norm)", f"{tag}: predicted energy per phoneme")
            details[sid] = {"text": sentence, "speaker": spk, "phones": phones,
                            "duration_ms": dur_ms.round(1).tolist(), "pitch_hz": pitch_hz.round(1).tolist(),
                            "energy": energy.round(3).tolist(), "total_s": round(float(dur_ms.sum()) / 1000, 2)}
    json.dump(details, open(os.path.join(a.out, "variance_predictions.json"), "w"), ensure_ascii=False, indent=1)

    log_dir = tc["path"]["log_path"]
    loss_plot(f"{log_dir}/train/log.txt", os.path.join(a.out, "training_loss.png"), f"{name}: training loss")
    loss_plot(f"{log_dir}/val/log.txt", os.path.join(a.out, "validation_loss.png"), f"{name}: validation loss")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["en", "te"], required=True)
    p.add_argument("--restore_step", type=int, required=True)
    p.add_argument("--speakers", nargs="*", default=None, help="speaker names from speakers.json; first one is plotted")
    p.add_argument("--out", required=True)
    main(p.parse_args())
