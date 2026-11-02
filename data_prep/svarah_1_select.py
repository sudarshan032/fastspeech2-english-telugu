# Stage 1E: Svarah -> raw_data/Svarah/<speaker>/<clip>.{wav,lab} at 22.05 kHz, 117 speakers, 5 h train / 1 h val.
import io, os, re, sys, glob, json, csv, random, subprocess, collections
import numpy as np, soundfile as sf, pyarrow.parquet as pq
from scipy.signal import resample_poly
from scipy.io import wavfile

subprocess.run("pip install -q inflect unidecode", shell=True, check=True)
subprocess.run("cd /tmp && git clone -q https://github.com/sudarshan032/FastSpeech2.git", shell=True, check=True)
sys.path.insert(0, "/tmp/FastSpeech2")
from text.cleaners import english_cleaners

OUT = "/kaggle/working"
RAW = f"{OUT}/raw_data/Svarah"
SR, MAXV = 22050, 32768.0
MIN_S, MAX_S = 1.0, 11.0
VAL_H, TRAIN_H = 1.0, 5.0
SEED = 1234
META = ["gender", "age-group", "primary_language", "native_place_state", "native_place_district",
        "highest_qualification", "job_category", "occupation_domain"]
# metadata groups whose clips looked mixed by voice (fs2-data-check, within-cos < 0.80). Prep v2 showed:
# South 24 Parganas = one person (2-means halves 0.98 similar) -> kept whole;
# Kolkata = one person + a few odd clips (minority cluster not self-consistent) -> drop the minority clips.
CHECK = {"Male|30-45|Bengali|West Bengal|Kolkata|Graduate|Freelance|Entertainment",
         "Male|60+|Bengali|West Bengal|South 24 Parganas|Post Graduate|Other|Financial Services (Banking and Insurance)"}
DROP_MINORITY = {"Male|30-45|Bengali|West Bengal|Kolkata|Graduate|Freelance|Entertainment"}
DIGITS = "zero one two three four five six seven eight nine".split()

SYMBOLS = [("%", " percent "), ("&", " and "), ("+", " plus "), ("@", " at "), ("_", " "), ("\r", " "), ("\n", " ")]


def clean(t):
    for a, b in SYMBOLS:
        t = t.replace(a, b)
    t = re.sub(r"\b[A-Z]{2,4}\b", lambda m: " ".join(m.group()), t)  # SBI -> S B I (spoken as letters)
    t = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1 to \2", t)  # 15-17 -> 15 to 17
    t = re.sub(r"\d{6,}", lambda m: " ".join(DIGITS[int(c)] for c in m.group()), t)  # phone/pincode digit by digit
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)  # 1,000 -> 1000 before number expansion
    return english_cleaners(t)


files = sorted(glob.glob("/kaggle/input/**/svarah_raw/data/*.parquet", recursive=True))
assert len(files) == 3, files
rows = []
for f in files:
    for r in pq.read_table(f).to_pylist():
        wav, sr = sf.read(io.BytesIO(r["audio_filepath"]["bytes"]), dtype="float32")
        d = len(wav) / sr
        if MIN_S <= d <= MAX_S:
            rows.append({"wav": wav, "sr": sr, "dur": d, "text": r["text"], "group": "|".join(str(r[c]) for c in META),
                         "src": r["audio_filepath"]["path"]})
print("clips 1-11 s:", len(rows), "hours:", round(sum(r["dur"] for r in rows) / 3600, 3), flush=True)

# split the two mixed groups by voice: WavLM-SV embeddings + 2-means
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
svm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
split_report = {}
for g in CHECK:
    idx = [i for i, r in enumerate(rows) if r["group"] == g]
    E = []
    with torch.no_grad():
        for i in idx:
            x = fe([rows[i]["wav"][: 16000 * 8]], sampling_rate=16000, return_tensors="pt")
            E.append(torch.nn.functional.normalize(svm(**x).embeddings, dim=-1)[0].numpy())
    E = np.stack(E)
    km = KMeans(2, n_init=10, random_state=SEED).fit(E)
    lab = km.labels_
    within = [float((E[lab == k] @ E[lab == k].T)[np.triu_indices((lab == k).sum(), 1)].mean()) for k in (0, 1)]
    split_report[g] = {"sizes": np.bincount(lab).tolist(), "silhouette": round(float(silhouette_score(E, lab, metric="cosine")), 3),
                       "within_cos_after": [round(w, 3) for w in within],
                       "between_centroid_cos": round(float(np.dot(*[km.cluster_centers_[k] / np.linalg.norm(km.cluster_centers_[k]) for k in (0, 1)])), 3)}
    split_report[g]["action"] = "drop minority clips" if g in DROP_MINORITY else "keep as one speaker"
    if g in DROP_MINORITY:
        minority = int(np.argmin(np.bincount(lab)))
        for i, k in zip(idx, lab):
            if k == minority:
                rows[i]["drop"] = True
rows = [r for r in rows if not r.get("drop")]
print("voice check:", json.dumps(split_report, indent=1), flush=True)

# speaker ids: spk000.. ordered by gender then metadata (stable)
groups = sorted({r["group"] for r in rows})
spk = {g: f"spk{i:03d}" for i, g in enumerate(groups)}
for r in rows:
    r["spk"] = spk[r["group"]]
by = collections.defaultdict(list)
for i, r in enumerate(rows):
    by[r["spk"]].append(i)

# split: val ~1 h from speakers with >= 3 min (proportional to their audio); train = the rest.
# train is then trimmed to ~5 h by dropping random clips of those same speakers (proportional, keeps the
# natural spread of minutes per speaker for the data-amount experiment); speakers < 3 min keep every clip.
rng = random.Random(SEED)
mins = {s: sum(rows[i]["dur"] for i in v) / 60 for s, v in by.items()}
big = [s for s in by if mins[s] >= 3.0]
share = VAL_H * 60 / sum(mins[s] for s in big)
val = set()
for s in big:
    ids = by[s][:]; rng.shuffle(ids)
    target, acc = mins[s] * share * 60, 0.0
    for i in ids:
        if acc >= target:
            break
        val.add(i); acc += rows[i]["dur"]
train = {i: s for s, v in by.items() for i in v if i not in val}
total = sum(rows[i]["dur"] for i in train)
cand = [i for i, s in train.items() if s in big]
rng.shuffle(cand)
dropped = []
for i in cand:
    if total <= TRAIN_H * 3600:
        break
    total -= rows[i]["dur"]; del train[i]; dropped.append(i)
tsec = collections.defaultdict(float)
for i, s in train.items():
    tsec[s] += rows[i]["dur"]

# write raw_data in the repo's layout + lists
os.makedirs(RAW, exist_ok=True)
lists = {"train": [], "val": []}
for split, ids in (("train", sorted(train)), ("val", sorted(val))):
    for i in ids:
        r = rows[i]
        base = f"{r['spk']}_{i:05d}"
        os.makedirs(f"{RAW}/{r['spk']}", exist_ok=True)
        w = resample_poly(r["wav"], 441, 320) if r["sr"] == 16000 else r["wav"]
        w = w / max(abs(w).max(), 1e-8) * MAXV * 0.95
        wavfile.write(f"{RAW}/{r['spk']}/{base}.wav", SR, w.astype(np.int16))
        t = clean(r["text"])
        open(f"{RAW}/{r['spk']}/{base}.lab", "w").write(t)
        lists[split].append(f"{base}|{r['spk']}|{r['dur']:.3f}|{t}|{r['text']}|{r['src']}")
for split in lists:
    open(f"{OUT}/{split}_list.txt", "w").write("\n".join(lists[split]) + "\n")

with open(f"{OUT}/speakers.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["speaker", *META, "minutes_1_11s", "train_min", "val_min"])
    for g in groups:
        s = spk[g]
        vm = sum(rows[i]["dur"] for i in by[s] if i in val) / 60
        w.writerow([s, *g.split("|"), round(mins[s], 2), round(tsec[s] / 60, 2), round(vm, 2)])

tm = sorted(tsec[s] / 60 for s in tsec)
summary = {
    "speakers": len(groups),
    "train": {"clips": len(train), "hours": round(sum(rows[i]["dur"] for i in train) / 3600, 3)},
    "val": {"clips": len(val), "hours": round(sum(rows[i]["dur"] for i in val) / 3600, 3), "speakers": len(big)},
    "unused_after_trim": {"clips": len(dropped), "hours": round(sum(rows[i]["dur"] for i in dropped) / 3600, 3)},
    "overlap_train_val": len(set(train) & val),
    "train_minutes_per_speaker": {"min": round(tm[0], 2), "median": round(tm[len(tm) // 2], 2), "max": round(tm[-1], 2)},
    "top_speaker_train_min": max(tsec.items(), key=lambda x: x[1])[0],
    "voice_check": split_report,
    "text_examples": [(r["text"], clean(r["text"])) for r in rows if re.search(r"\d{6,}|\d\s*-\s*\d|\b[A-Z]{2,4}\b|%|&", r["text"])][:20],
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))
