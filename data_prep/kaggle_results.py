# Stage 5 on CPU: test sentences + variance-adaptor plots + loss curves from a training kernel's output.
import os, glob, shutil, subprocess
def sh(c):
    print("+", c, flush=True); return subprocess.run(c, shell=True).returncode
OUT, REPO = "/kaggle/working", "/kaggle/working/FastSpeech2"
sh(f"git clone -q https://github.com/sudarshan032/FastSpeech2.git {REPO} && cd {REPO}/hifigan && unzip -q -o generator_universal.pth.tar.zip")
sh("pip install -q tgt pyworld g2p-en inflect unidecode pypinyin")
sh("python3 -c \"import nltk; [nltk.download(p, quiet=True) for p in ('averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'cmudict')]\"")
JOBS = [("IndicTTS_Telugu", "te", "")]
for name, lang, extra in JOBS:
    pp = glob.glob(f"/kaggle/input/**/preprocessed_data_{name}", recursive=True)[0]
    os.makedirs(f"{REPO}/preprocessed_data/{name}", exist_ok=True)
    for f in ("stats.json", "speakers.json"):
        shutil.copy(f"{pp}/{f}", f"{REPO}/preprocessed_data/{name}/")
    for lex in glob.glob("/kaggle/input/**/*_lexicon.txt", recursive=True):
        shutil.copy(lex, f"{REPO}/lexicon/")
    ck = glob.glob(f"/kaggle/input/**/ckpt/{name}/*.pth.tar", recursive=True)
    step = max(int(os.path.basename(p).split(".")[0]) for p in ck)
    os.makedirs(f"{REPO}/output/ckpt/{name}", exist_ok=True)
    shutil.copy([p for p in ck if os.path.basename(p) == f"{step}.pth.tar"][0], f"{REPO}/output/ckpt/{name}/")
    logs = glob.glob(f"/kaggle/input/**/log/{name}", recursive=True)
    if logs:
        shutil.copytree(logs[0], f"{REPO}/output/log/{name}")
    sh(f"cd {REPO} && python3 make_results.py --lang {lang} --restore_step {step} {extra} --out {OUT}/Results/{name} 2>&1 | grep -v Warning | tail -60")
sh(f"mkdir -p {OUT}/meta && python3 --version > {OUT}/meta/versions.txt && pip list 2>/dev/null | grep -iE '^(torch|numpy|scipy|librosa|pyworld|tgt|g2p-en|inflect|unidecode|matplotlib|pyyaml|tqdm|tensorboard|soundfile|nltk|pypinyin|scikit-learn|numba) ' >> {OUT}/meta/versions.txt")
# small files for the submission repo (listing the prep kernels' outputs directly hits API rate limits)
import torch
for name in ("Svarah", "IndicTTS_Telugu"):
    M = f"{OUT}/meta/{name}"
    os.makedirs(M, exist_ok=True)
    pp = glob.glob(f"/kaggle/input/**/preprocessed_data_{name}", recursive=True)[0]
    for f in ("stats.json", "speakers.json", "train.txt", "val.txt", "train_ids.txt", "val_ids.txt"):
        for p in glob.glob(f"{pp}/{f}") + glob.glob(f"{os.path.dirname(pp)}/{f}"):
            shutil.copy(p, M)
    ck = glob.glob(f"/kaggle/input/**/ckpt/{name}/*.pth.tar", recursive=True)
    step = max(int(os.path.basename(p).split(".")[0]) for p in ck)
    c = torch.load([p for p in ck if os.path.basename(p) == f"{step}.pth.tar"][0], map_location="cpu", weights_only=False)
    torch.save({"model": c["model"]}, f"{OUT}/meta/{name}/{step}.pth.tar")  # model weights only (no optimizer state)
for lex in glob.glob("/kaggle/input/**/*_lexicon.txt", recursive=True) + glob.glob("/kaggle/input/**/*mfa*.zip", recursive=True) + glob.glob("/kaggle/input/**/*.dict", recursive=True) + glob.glob("/kaggle/input/**/drop_report.json", recursive=True):
    shutil.copy(lex, f"{OUT}/meta/")
shutil.rmtree(REPO, ignore_errors=True)
sh(f"find {OUT}/Results | sort")
