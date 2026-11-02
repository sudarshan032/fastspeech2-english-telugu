# Stages 1T-3T: Telugu male speaker -> 5 h / 1 h split, 22.05 kHz wavs, rule-based lexicon,
# MFA acoustic model trained on our data (probe on 300 clips first), FastSpeech2 preprocessing.
# Nothing here raises: each stage logs its result, so the outputs of finished stages are always saved.
import io, os, re, sys, glob, json, random, shutil, subprocess, collections, traceback
from multiprocessing import Pool
import numpy as np, soundfile as sf, pyarrow.parquet as pq
from scipy.signal import resample_poly
from scipy.io import wavfile

OUT, W, J = "/kaggle/working", "/tmp/w", os.cpu_count()
REPO = f"{OUT}/FastSpeech2"
SR, MIN_S, MAX_S, VAL_H, TRAIN_H, SEED = 22050, 1.0, 11.0, 1.0, 5.0, 1234
SPK = "te_male"

def sh(c):
    print("+", c, flush=True)
    r = subprocess.run(c, shell=True).returncode
    print("returncode", r, flush=True)
    return r

install = subprocess.Popen("cd /tmp && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba"
                           " && /tmp/bin/micromamba create -q -y -p /tmp/mfa -c conda-forge montreal-forced-aligner python=3.11", shell=True)
sh("pip install -q tgt pyworld inflect unidecode")
sh(f"git clone -q https://github.com/sudarshan032/FastSpeech2.git {REPO}")
sys.path.insert(0, REPO)
from text import telugu

# ---- 1T: select male clips 1-11 s
rows = []
for f in sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True)):
    for r in pq.read_table(f).to_pylist():
        g = r["gender"]
        if not (g == 1 or str(g).lower() == "male"):
            continue
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        d = len(wav) / sr
        ws = telugu.words(r["text"])
        if MIN_S <= d <= MAX_S and ws:
            rows.append({"wav": wav, "sr": sr, "dur": d, "orig": r["text"], "words": ws})
print("male clips 1-11 s:", len(rows), "hours", round(sum(r["dur"] for r in rows) / 3600, 3),
      "sample rates", collections.Counter(r["sr"] for r in rows), flush=True)

# split by sentence (identical sentences stay in one split), ~1 h val, train trimmed to 5 h
groups = collections.defaultdict(list)
for i, r in enumerate(rows):
    groups[" ".join(r["words"])].append(i)
keys = sorted(groups); random.Random(SEED).shuffle(keys)
val, train, unused, acc = [], [], [], 0.0
for k in keys:
    gd = sum(rows[i]["dur"] for i in groups[k])
    if acc < VAL_H * 3600:
        val += groups[k]; acc += gd
    elif sum(rows[i]["dur"] for i in train) + gd <= TRAIN_H * 3600:  # ponytail: O(n^2) sum, n ~ 3k
        train += groups[k]
    else:
        unused += groups[k]

RAW = f"{OUT}/raw_data/IndicTTS_Telugu/{SPK}"
os.makedirs(RAW, exist_ok=True)
def write(job):
    i, base = job
    r = rows[i]
    w = resample_poly(r["wav"], SR // 150, r["sr"] // 150) if r["sr"] != SR else r["wav"]  # 48k: 147/320
    w = w / max(abs(w).max(), 1e-8) * 32768.0 * 0.95
    wavfile.write(f"{RAW}/{base}.wav", SR, w.astype(np.int16))
    open(f"{RAW}/{base}.lab", "w", encoding="utf-8").write(" ".join(r["words"]))
names = {i: f"te_{i:05d}" for i in train + val}
with Pool(J) as pool:
    pool.map(write, list(names.items()), chunksize=16)
for s, ids in (("train", train), ("val", val)):
    with open(f"{OUT}/{s}_list.txt", "w", encoding="utf-8") as f:
        for i in sorted(ids):
            f.write(f"{names[i]}|{rows[i]['dur']:.3f}|{' '.join(rows[i]['words'])}|{rows[i]['orig'].replace(chr(10), ' ')}\n")
lex = {}
for i in names:
    for w in rows[i]["words"]:
        lex.setdefault(w, telugu.word_to_phones(w))
with open(f"{OUT}/telugu_lexicon.txt", "w", encoding="utf-8") as f:
    for w in sorted(lex):
        f.write(f"{w}\t{' '.join(lex[w])}\n")
summary = {"selected": {"clips": len(rows), "hours": round(sum(r["dur"] for r in rows) / 3600, 3)},
           "train": {"clips": len(train), "hours": round(sum(rows[i]["dur"] for i in train) / 3600, 3)},
           "val": {"clips": len(val), "hours": round(sum(rows[i]["dur"] for i in val) / 3600, 3)},
           "unused": {"clips": len(unused), "hours": round(sum(rows[i]["dur"] for i in unused) / 3600, 3)},
           "sentence_overlap_train_val": len({" ".join(rows[i]["words"]) for i in train} & {" ".join(rows[i]["words"]) for i in val}),
           "lexicon_words": len(lex), "phones_used": sorted({p for v in lex.values() for p in v}),
           "examples": [(rows[i]["orig"], " ".join(rows[i]["words"]), " | ".join(" ".join(lex[w]) for w in rows[i]["words"])) for i in val[:5]]}
print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
del rows

# ---- 2T: MFA acoustic model trained on this data (no pretrained Telugu model in MFA)
install.wait()
sh(f"useradd -m mfauser; mkdir -p {W}/mfa_root {W}/small/{SPK}")
for b in sorted(names.values())[:300]:  # probe corpus
    for ext in ("wav", "lab"):
        shutil.copy(f"{RAW}/{b}.{ext}", f"{W}/small/{SPK}/{b}.{ext}")
shutil.copytree(f"{OUT}/raw_data/IndicTTS_Telugu", f"{W}/corpus")
shutil.copy(f"{OUT}/telugu_lexicon.txt", f"{W}/lexicon.txt")
sh(f"chown -R mfauser {W}")
mfa = f"runuser -u mfauser -- /usr/bin/env HOME=/home/mfauser MFA_ROOT_DIR={W}/mfa_root PATH=/tmp/mfa/bin:/usr/bin:/bin mfa"
sh(f"{mfa} version")
probe_rc = sh(f"{mfa} train {W}/small {W}/lexicon.txt {W}/small_model.zip --output_directory {W}/small_tg -j {J} --clean")
n_small = len(glob.glob(f"{W}/small_tg/**/*.TextGrid", recursive=True))
print("PROBE: returncode", probe_rc, "textgrids", n_small, "of 300", flush=True)
if os.path.exists(f"{W}/small_tg"):
    shutil.copytree(f"{W}/small_tg", f"{OUT}/probe_TextGrid")
if probe_rc == 0 and n_small > 250:
    rc = sh(f"{mfa} train {W}/corpus {W}/lexicon.txt {W}/telugu_mfa_model.zip --output_directory {W}/TextGrid -j {J} --clean")
    if os.path.exists(f"{W}/TextGrid"):
        shutil.copytree(f"{W}/TextGrid", f"{OUT}/TextGrid")
    if os.path.exists(f"{W}/telugu_mfa_model.zip"):
        shutil.copy(f"{W}/telugu_mfa_model.zip", OUT)
    print("FULL: returncode", rc, "textgrids", len(glob.glob(f"{OUT}/TextGrid/**/*.TextGrid", recursive=True)), "of", len(names), flush=True)

# ---- 3T: FastSpeech2 preprocessing
try:
    if os.path.exists(f"{OUT}/TextGrid"):
        PP = f"{REPO}/preprocessed_data/IndicTTS_Telugu"
        shutil.copytree(f"{OUT}/TextGrid", f"{PP}/TextGrid", ignore=shutil.ignore_patterns("*.csv"))
        os.makedirs(f"{REPO}/raw_data", exist_ok=True)
        os.symlink(f"{OUT}/raw_data/IndicTTS_Telugu", f"{REPO}/raw_data/IndicTTS_Telugu")
        for s in ("train", "val"):
            ids = [l.split("|")[0] for l in open(f"{OUT}/{s}_list.txt", encoding="utf-8")]
            open(f"{PP}/{s}_ids.txt", "w").write("\n".join(ids) + "\n")
        os.chdir(REPO)
        sh("python3 preprocess.py config/IndicTTS_Telugu/preprocess.yaml")
        for f in ("train.txt", "val.txt"):
            print(f, sum(1 for _ in open(f"{PP}/{f}")), flush=True)
        print(open(f"{PP}/stats.json").read(), open(f"{PP}/train.txt").readline(), flush=True)
        shutil.move(PP, f"{OUT}/preprocessed_data_IndicTTS_Telugu")
except Exception:
    traceback.print_exc()
os.chdir(OUT)
shutil.rmtree(REPO, ignore_errors=True)
print("done", flush=True)
