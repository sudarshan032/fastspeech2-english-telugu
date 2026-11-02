"""Telugu: synthesize the assignment test sentences (or --text) and write audio + variance-adaptor plots.
Usage: python Telugu/inference.py --restore_step N [--text "..."] [--out ../Results/Telugu]"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fastspeech2"))
import run

p = argparse.ArgumentParser()
p.add_argument("--restore_step", type=int, required=True)
p.add_argument("--text", default=None, help="a custom sentence instead of the test set")
p.add_argument("--speaker", default=None, help="speaker name from speakers.json (English only)")
p.add_argument("--out", default=os.path.join(HERE, "..", "Results", "Telugu"))
a = p.parse_args()
run.inference(os.path.join(HERE, "config.yaml"), "te", a.restore_step, os.path.abspath(a.out),
              speakers=None, text=a.text, speaker=a.speaker)
