# Stage 3E (part 2): final English clip lists + FastSpeech2 preprocessing (mel, pitch, energy, duration).
import os, re, glob, json, shutil, subprocess, collections

def sh(c):
    print("+", c, flush=True)
    subprocess.run(c, shell=True, check=True)

OUT, REPO = "/kaggle/working", "/kaggle/working/FastSpeech2"
sh("pip install -q tgt pyworld g2p-en inflect unidecode")
sh(f"git clone -q https://github.com/sudarshan032/FastSpeech2.git {REPO}")
import tgt

raw = glob.glob("/kaggle/input/**/raw_data/Svarah", recursive=True)[0]
lists = os.path.dirname(os.path.dirname(raw))
tg_src = [p for p in glob.glob("/kaggle/input/**/TextGrid", recursive=True) if os.path.isdir(p)][0]
lex_src = glob.glob("/kaggle/input/**/svarah_lexicon.txt", recursive=True)[0]
PP = f"{REPO}/preprocessed_data/Svarah"
os.makedirs(f"{REPO}/raw_data", exist_ok=True)
os.symlink(raw, f"{REPO}/raw_data/Svarah")
shutil.copytree(tg_src, f"{PP}/TextGrid", ignore=shutil.ignore_patterns("*.csv"))
shutil.copy(lex_src, f"{REPO}/lexicon/svarah_lexicon.txt")

# clip lists from data selection
split, dur = {}, {}
for s in ("train", "val"):
    for rec in re.split(r"\n(?=spk\d{3}_\d{5}\|)", open(f"{lists}/{s}_list.txt", encoding="utf-8").read().strip()):
        base, spk, d, _ = rec.split("|", 3)
        split[base], dur[base] = s, float(d)

# drop rules (from the 20-clip MFA probe):
#  - "Rs": written "Rs" is spoken "rupees" (maybe after the number) -> text/audio mismatch
#  - spelled-out acronyms where MFA squeezed a letter below 50 ms (PIN, PRAN, AM said as one word)
drops = collections.defaultdict(list)
for base in split:
    p = f"{PP}/TextGrid/{base[:6]}/{base}.TextGrid"
    if not os.path.exists(p):
        drops["alignment_failed"].append(base); continue
    words = [(w.text, w.end_time - w.start_time) for w in tgt.io.read_textgrid(p).get_tier_by_name("words")._objects if w.text]
    if any(t == "rs" for t, _ in words):
        drops["rs"].append(base); continue
    run, bad = [], False
    for t, d in words + [("<end>", 1)]:
        if len(t) == 1 and t.isalpha():
            run.append(d)
        else:
            bad |= len(run) >= 2 and min(run) < 0.05
            run = []
    if bad:
        drops["squeezed_letters"].append(base)
dropped = {b for v in drops.values() for b in v}
for s in ("train", "val"):
    open(f"{PP}/{s}_ids.txt", "w").write("\n".join(b for b in split if split[b] == s and b not in dropped) + "\n")
report = {k: {"clips": len(v), "minutes": round(sum(dur[b] for b in v) / 60, 2), "examples": v[:15]} for k, v in drops.items()}
for s in ("train", "val"):
    kept = [b for b in split if split[b] == s and b not in dropped]
    report[f"{s}_kept"] = {"clips": len(kept), "hours": round(sum(dur[b] for b in kept) / 3600, 3)}
print(json.dumps(report, indent=1), flush=True)
json.dump(report, open(f"{OUT}/drop_report.json", "w"), indent=1)

os.chdir(REPO)
sh("python3 preprocess.py config/Svarah/preprocess.yaml")
for f in ("train.txt", "val.txt", "speakers.json", "stats.json"):
    print(f, sum(1 for _ in open(f"{PP}/{f}")) if f.endswith("txt") else open(f"{PP}/{f}").read()[:300])
print(open(f"{PP}/train.txt").readline())
# keep only the data products (repo is re-cloned by the training kernel)
shutil.move(PP, f"{OUT}/preprocessed_data_Svarah")
shutil.copy(f"{REPO}/lexicon/svarah_lexicon.txt", OUT)
shutil.rmtree(REPO)
print("done", flush=True)
