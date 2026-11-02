# Stage 3E (part 1): MFA alignment of raw_data/Svarah + per-clip fit scores. Nothing is dropped here.
import os, re, glob, csv, json, shutil, subprocess, collections
import numpy as np

def sh(c, env=None):
    print("+", c, flush=True)
    subprocess.run(c, shell=True, check=True, env=env)

OUT = "/kaggle/working"
src = glob.glob("/kaggle/input/**/raw_data/Svarah", recursive=True)[0]
lists = os.path.dirname(os.path.dirname(src))
W = "/tmp/w"
J = os.cpu_count()  # everything MFA touches lives here, owned by a non-root user
CORPUS = f"{W}/corpus"
shutil.copytree(src, CORPUS)  # MFA wants a writable corpus folder
print("corpus files:", sum(len(f) for _, _, f in os.walk(CORPUS)), flush=True)

# MFA from conda-forge via micromamba (not on Kaggle by default)
sh("cd /tmp && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba")
sh("/tmp/bin/micromamba create -q -y -p /tmp/mfa -c conda-forge montreal-forced-aligner python=3.11")
# MFA 3 runs a local PostgreSQL server, which refuses to start as root (Kaggle runs as root) -> run MFA as a normal user
sh(f"useradd -m mfauser && mkdir -p {W}/mfa_root && chown -R mfauser {W}")
env = None
mfa = f"runuser -u mfauser -- /usr/bin/env HOME=/home/mfauser MFA_ROOT_DIR={W}/mfa_root PATH=/tmp/mfa/bin:/usr/bin:/bin mfa"
sh(f"{mfa} version")
for kind in ("acoustic", "dictionary", "g2p"):
    sh(f"{mfa} model download {kind} english_us_arpa")
DICT = glob.glob(f"{W}/mfa_root/pretrained_models/dictionary/english_us_arpa*.dict")[0]

# unknown words -> pronunciations from MFA's pretrained English G2P, merged into a corpus dictionary
sh(f"{mfa} find_oovs {CORPUS} {DICT} {W}/oov -j {J} --clean")
oov_file = (glob.glob(f"{W}/oov/**/oovs_found*.txt", recursive=True) + [None])[0]
oovs = [w.strip() for w in open(oov_file)] if oov_file else []
oovs = [w for w in oovs if w]
print("OOV words:", len(oovs), oovs[:60], flush=True)
merged = f"{W}/svarah.dict"
shutil.copy(DICT, merged)
if oovs:
    open(f"{W}/oovs.txt", "w").write("\n".join(oovs) + "\n")
    # probe showed g2p returns up to 10 variants per word in no useful order -> ask for the single best one
    g2p = f"chown -R mfauser {W} && {mfa} g2p {W}/oovs.txt english_us_arpa {W}/oov_prons.txt -j {J} --clean"
    try:
        sh(g2p + " --num_pronunciations 1")
    except subprocess.CalledProcessError:  # flag not supported -> all variants, keep the first per word
        sh(g2p)
        seen, keep = set(), []
        for l in open(f"{W}/oov_prons.txt"):
            if l.split("\t")[0] not in seen:
                seen.add(l.split("\t")[0]); keep.append(l)
        open(f"{W}/oov_prons.txt", "w").writelines(keep)
    with open(merged, "a") as f:
        f.write(open(f"{W}/oov_prons.txt").read())

# align
sh(f"chown -R mfauser {W} && {mfa} align {CORPUS} {merged} english_us_arpa {W}/TextGrid -j {J} --clean")
shutil.copytree(f"{W}/TextGrid", f"{OUT}/TextGrid")
shutil.copy(merged, f"{OUT}/svarah_mfa.dict")

# word -> ARPAbet lexicon for synthesize.py: one pronunciation per word, the most probable one.
# pretrained dict rows: word, prob, 3 silence probs, phones; g2p rows: word, phones (treated as prob 1)
lex = {}
for line in open(merged, encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2 or not parts[0]:
        continue
    prob = float(parts[1]) if len(parts) == 6 else 1.0
    if parts[0] not in lex or prob > lex[parts[0]][0]:
        lex[parts[0]] = (prob, parts[-1].split())
lex = {w: p for w, (_, p) in lex.items()}
with open(f"{OUT}/svarah_lexicon.txt", "w") as f:
    for w, p in lex.items():
        f.write(f"{w}\t{' '.join(p)}\n")

print("textgrids:", len(glob.glob(f"{OUT}/TextGrid/**/*.TextGrid", recursive=True)), "lexicon words:", len(lex), flush=True)
print("g2p variants per word (should be 1):", collections.Counter(collections.Counter(l.split("\t")[0] for l in open(f"{W}/oov_prons.txt")).values()) if oovs else None)
