"""English: train FastSpeech2 from scratch.
Usage: python English/train.py [--restore_step N] [--max_hours H]"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fastspeech2"))
import run

p = argparse.ArgumentParser()
p.add_argument("--restore_step", type=int, default=0)
p.add_argument("--max_hours", type=float, default=0, help="stop and save after this many hours (0 = no limit)")
a = p.parse_args()
run.train(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"), a.restore_step, a.max_hours)
