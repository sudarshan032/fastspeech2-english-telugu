# English: validation loss per speaker vs that speaker's training minutes (final checkpoint, teacher-forced
# exactly like evaluate.py). Also train-set loss with dropout off on up to 15 clips per speaker.
import os, glob, json, time, shutil, subprocess, collections
from concurrent.futures import ThreadPoolExecutor
def sh(c):
    print("+", c, flush=True); return subprocess.run(c, shell=True).returncode
OUT, REPO = "/kaggle/working", "/kaggle/working/FastSpeech2"
sh(f"git clone -q https://github.com/sudarshan032/FastSpeech2.git {REPO}")
sh("pip install -q tgt pyworld g2p-en inflect unidecode pypinyin")
import sys; os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from types import SimpleNamespace
from dataset import Dataset
from model import FastSpeech2Loss
from utils.model import get_model
from utils.tools import to_device
torch.set_num_threads(os.cpu_count())

src = glob.glob("/kaggle/input/**/preprocessed_data_Svarah", recursive=True)[0]
PP = f"{REPO}/preprocessed_data/Svarah"; os.makedirs(PP, exist_ok=True)
for d in ("mel", "pitch", "energy", "duration"):
    os.symlink(f"{src}/{d}", f"{PP}/{d}")
for f in ("train.txt", "val.txt", "stats.json", "speakers.json"):
    shutil.copy(f"{src}/{f}", PP)
ck = glob.glob("/kaggle/input/**/ckpt/Svarah/*.pth.tar", recursive=True)
step = max(int(os.path.basename(p).split(".")[0]) for p in ck)
os.makedirs(f"{REPO}/output/ckpt/Svarah", exist_ok=True)
shutil.copy([p for p in ck if os.path.basename(p) == f"{step}.pth.tar"][0], f"{REPO}/output/ckpt/Svarah/")
cfg = [yaml.load(open(f"config/Svarah/{k}.yaml"), Loader=yaml.FullLoader) for k in ("preprocess", "model", "train")]
cfg[2]["optimizer"]["batch_size"] = 1
model = get_model(SimpleNamespace(restore_step=step), cfg, "cpu", train=False)
Loss = FastSpeech2Loss(cfg[0], cfg[1])
stats = json.load(open(f"{PP}/stats.json"))

# training minutes per speaker from the duration targets (frames per phone, hop 256 @ 22,050 Hz)
lines = [l.split("|") for l in open(f"{PP}/train.txt", encoding="utf-8")]
def secs(l): return np.load(f"{PP}/duration/{l[1]}-duration-{l[0]}.npy").sum() * 256 / 22050
with ThreadPoolExecutor(32) as ex:
    train_sec = list(ex.map(secs, lines))
train_min = collections.Counter()
for l, s in zip(lines, train_sec):
    train_min[l[1]] += s / 60
# train subset: up to 15 clips per speaker
per = collections.defaultdict(list)
for l in lines:
    if len(per[l[1]]) < 15: per[l[1]].append("|".join(l))
open(f"{PP}/train_sub.txt", "w", encoding="utf-8").write("".join(x for v in per.values() for x in v))

NAMES = ["total", "mel", "postnet", "pitch", "energy", "duration"]
def run(fname, limit=None):
    ds = Dataset(fname, cfg[0], cfg[2], sort=False, drop_last=False)
    rows = []
    t = time.time()
    for batchs in DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_fn):
        for batch in batchs:
            spk = batch[0][0].split("_")[0]
            b = to_device(batch, "cpu")
            with torch.no_grad():
                losses = Loss(b, model(*(b[2:])))
            f0 = float(np.mean(batch[9][0][: batch[4][0]]) * stats["pitch"][3] + stats["pitch"][2])
            rows.append({"id": batch[0][0], "speaker": spk, "mel_frames": int(batch[7][0]), "f0_hz": f0,
                         **{n: float(v) for n, v in zip(NAMES, losses)}})
        if limit and len(rows) >= limit: break
    print(fname, len(rows), "clips in", round(time.time() - t, 1), "s", flush=True)
    return rows

# probe: 20 clips; the mean must sit near the logged validation loss (2.60 total, 0.757 mel) or something is off
probe = run("val.txt", 20)
print("probe mean", {n: round(np.mean([r[n] for r in probe]), 3) for n in NAMES}, flush=True)
val = run("val.txt")
tr = run("train_sub.txt")
m_all = {n: np.average([r[n] for r in val], weights=None) for n in NAMES}
print("full val mean (unweighted by clip)", {n: round(v, 4) for n, v in m_all.items()}, flush=True)

def by_spk(rows):
    g = collections.defaultdict(list)
    for r in rows: g[r["speaker"]].append(r)
    return {s: {**{n: float(np.mean([r[n] for r in v])) for n in NAMES}, "clips": len(v),
                "f0_hz": float(np.mean([r["f0_hz"] for r in v]))} for s, v in g.items()}
V, T = by_spk(val), by_spk(tr)
spks = sorted(V)
table = [{"speaker": s, "train_min": round(train_min[s], 2), "val_clips": V[s]["clips"], "f0_hz": round(V[s]["f0_hz"], 1),
          **{f"val_{n}": round(V[s][n], 4) for n in NAMES}, **{f"train_{n}": round(T[s][n], 4) for n in NAMES if s in T}}
         for s in spks]
os.makedirs(f"{OUT}/speaker_eval", exist_ok=True)
import csv
with open(f"{OUT}/speaker_eval/per_speaker.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
json.dump({"val": val, "train_sub": tr}, open(f"{OUT}/speaker_eval/per_clip.json", "w"))

mins = np.array([r["train_min"] for r in table])
summary = {"speakers_in_val": len(spks), "train_min_range": [float(mins.min()), float(np.median(mins)), float(mins.max())]}
for n in NAMES:
    rho, p = spearmanr(mins, [r[f"val_{n}"] for r in table])
    summary[f"spearman_trainmin_vs_val_{n}"] = {"rho": round(float(rho), 3), "p": float(p)}
    gap = [r[f"val_{n}"] - r[f"train_{n}"] for r in table if f"train_{n}" in r]
    rho2, p2 = spearmanr([r["train_min"] for r in table if f"train_{n}" in r], gap)
    summary[f"spearman_trainmin_vs_gap_{n}"] = {"rho": round(float(rho2), 3), "p": float(p2)}
bins = [(0, 3), (3, 5), (5, 7), (7, 100)]
summary["bins"] = {f"{a}-{b} min": {"speakers": int(((mins >= a) & (mins < b)).sum()),
                   **{n: round(float(np.mean([r[f"val_{n}"] for r in table if a <= r["train_min"] < b])), 4) for n in ("mel", "duration", "pitch", "energy")}}
                   for a, b in bins if ((mins >= a) & (mins < b)).any()}
summary["train_sub_mean_dropout_off"] = {n: round(float(np.mean([r[n] for r in tr])), 4) for n in NAMES}
summary["val_mean"] = {n: round(float(v), 4) for n, v in m_all.items()}
json.dump(summary, open(f"{OUT}/speaker_eval/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, n in zip(axes, ("mel", "duration", "pitch")):
    c = ["#c0392b" if r["f0_hz"] > 165 else "#2f56b0" for r in table]
    ax.scatter(mins, [r[f"val_{n}"] for r in table], c=c, s=18)
    ax.set_xlabel("training minutes of this speaker"); ax.set_ylabel(f"validation {n} loss")
    ax.set_title(f"{n}: Spearman rho = {summary[f'spearman_trainmin_vs_val_{n}']['rho']}"); ax.grid(alpha=.3)
fig.suptitle("English: per-speaker validation loss vs training data (red = mean F0 > 165 Hz, blue = lower)")
fig.tight_layout(); fig.savefig(f"{OUT}/speaker_eval/val_loss_vs_minutes.png", dpi=140)
shutil.rmtree(REPO, ignore_errors=True)
print("done")
