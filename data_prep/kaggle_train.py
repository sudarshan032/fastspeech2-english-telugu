# Stage 4: train English (GPU 0) and Telugu (GPU 1) at the same time in one T4x2 session.
# HOURS caps training time per model; each run saves a final checkpoint when time is up.
import os, glob, shutil, subprocess, time

HOURS = 6.0
OUT, REPO = "/kaggle/working", "/kaggle/working/FastSpeech2"

def sh(c):
    print("+", c, flush=True)
    return subprocess.run(c, shell=True).returncode

sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc")
sh(f"git clone -q https://github.com/sudarshan032/FastSpeech2.git {REPO} && cd {REPO} && git log --oneline | head -1")
sh(f"cd {REPO}/hifigan && unzip -q -o generator_universal.pth.tar.zip")
sh("pip install -q tgt pyworld g2p-en inflect unidecode pypinyin")
sh("python3 -c \"import nltk; [nltk.download(p, quiet=True) for p in ('averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'cmudict')]\"")
os.makedirs(f"{REPO}/preprocessed_data", exist_ok=True)
runs = {"Svarah": "0"}
procs = {}
for name, gpu in runs.items():
    src = glob.glob(f"/kaggle/input/**/preprocessed_data_{name}", recursive=True)
    if not src:
        print("missing preprocessed data for", name, flush=True)
        continue
    # dry run showed reading .npy files from /kaggle/input starves the GPU (~12 s per 64 clips) -> local copy
    # a plain copy took 32 min (per-file latency) -> copy with many threads
    from concurrent.futures import ThreadPoolExecutor
    files = [os.path.join(d, f) for d, _, fs in os.walk(src[0]) for f in fs]
    dst = f"{REPO}/preprocessed_data/{name}"
    for d, _, _ in os.walk(src[0]):
        os.makedirs(d.replace(src[0], dst, 1), exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(64) as ex:
        list(ex.map(lambda f: shutil.copyfile(f, f.replace(src[0], dst, 1)), files))
    print("copied", len(files), "files in", round(time.time() - t0), "s", flush=True)
    for lex in glob.glob("/kaggle/input/**/*_lexicon.txt", recursive=True):
        shutil.copy(lex, f"{REPO}/lexicon/")
    lines = open(f"{dst}/train.txt", encoding="utf-8").read().split("\n")
    print(name, "train utterances", len(lines) - 1, "| with internal pauses (sp):", sum(" sp " in l for l in lines), flush=True)
    cmd = (f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} python3 train.py --max_hours {HOURS} "
           f"-p config/{name}/preprocess.yaml -m config/{name}/model.yaml -t config/{name}/train.yaml")
    print("+", cmd, flush=True)
    procs[name] = subprocess.Popen(cmd, shell=True, stdout=open(f"{OUT}/train_{name}.log", "w"), stderr=subprocess.STDOUT)

# progress heartbeat into the kernel log every 10 min
while any(p.poll() is None for p in procs.values()):
    time.sleep(60 if HOURS < 1 else 600)
    for name in procs:
        logs = glob.glob(f"{REPO}/output/log/{name}/*/log.txt")
        last = [open(l).read().strip().split("\n")[-1][:220] for l in logs if os.path.getsize(l)]
        print(time.strftime("%H:%M"), name, "rc", procs[name].poll(), last, flush=True)
for name, p in procs.items():
    print(name, "exit code", p.returncode, flush=True)
    print(open(f"{OUT}/train_{name}.log").read()[-3000:], flush=True)

# Stage 5: test sentences, variance-adaptor plots, loss curves (final checkpoint of each model)
import csv
voices = list(csv.DictReader(open(glob.glob("/kaggle/input/**/speakers.csv", recursive=True)[0])))
main_v = max(voices, key=lambda r: float(r["train_min"]))
other_v = max((r for r in voices if r["gender"] != main_v["gender"]), key=lambda r: float(r["train_min"]))
print("voices:", main_v["speaker"], main_v["gender"], main_v["train_min"], "|", other_v["speaker"], other_v["gender"], other_v["train_min"], flush=True)
for name, lang, extra in (("Svarah", "en", f"--speakers {main_v['speaker']} {other_v['speaker']}"), ("IndicTTS_Telugu", "te", "")):
    ck = glob.glob(f"{REPO}/output/ckpt/{name}/*.pth.tar")
    if not ck:
        continue
    step = max(int(os.path.basename(p).split(".")[0]) for p in ck)
    sh(f"cd {REPO} && python3 make_results.py --lang {lang} --restore_step {step} {extra} --out {OUT}/Results/{name} 2>&1 | grep -v Warning | tail -40")

# keep: final checkpoint (with optimizer, for resuming) + every 50k-step checkpoint (model only)
for name in procs:
    ck = sorted(glob.glob(f"{REPO}/output/ckpt/{name}/*.pth.tar"), key=lambda p: int(os.path.basename(p).split(".")[0]))
    dst = f"{OUT}/ckpt/{name}"; os.makedirs(dst, exist_ok=True)
    for i, p in enumerate(ck):
        step = int(os.path.basename(p).split(".")[0])
        if i == len(ck) - 1:
            shutil.copy(p, dst)
        elif step % 50000 == 0:
            import torch
            torch.save({"model": torch.load(p, map_location="cpu", weights_only=False)["model"]}, f"{dst}/{step}.pth.tar")
    if os.path.exists(f"{REPO}/output/log/{name}"):
        shutil.copytree(f"{REPO}/output/log/{name}", f"{OUT}/log/{name}")
shutil.rmtree(REPO, ignore_errors=True)
print("done", flush=True)
