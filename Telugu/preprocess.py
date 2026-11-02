"""Telugu: extract mel, pitch, energy and durations (needs raw_data/ and MFA TextGrids, see README).
Usage: python Telugu/preprocess.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fastspeech2"))
import run

run.preprocess(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
