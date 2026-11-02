"""Shared entry points for English/ and Telugu/: one config.yaml per language holding the
preprocess, model and train sections of the original FastSpeech2 code."""
import os
import sys
from types import SimpleNamespace

import yaml

CODE = os.path.dirname(os.path.abspath(__file__))


def load(config_path):
    cfg = yaml.load(open(config_path), Loader=yaml.FullLoader)
    os.chdir(CODE)  # data/lexicon paths in config.yaml are relative to this folder
    sys.path.insert(0, CODE)
    return cfg["preprocess"], cfg["model"], cfg["train"]


def preprocess(config_path):
    pp, _, _ = load(config_path)
    from preprocessor.preprocessor import Preprocessor

    Preprocessor(pp).build_from_path()


def train(config_path, restore_step=0, max_hours=0):
    configs = load(config_path)
    import train as fs2_train

    fs2_train.main(SimpleNamespace(restore_step=restore_step, max_hours=max_hours), configs)


def inference(config_path, lang, restore_step, out_dir, speakers=None, text=None, speaker=None):
    configs = load(config_path)
    import make_results

    if text is None:  # the assignment's test sentences + variance-adaptor plots + loss curves
        make_results.main(
            SimpleNamespace(lang=lang, restore_step=restore_step, speakers=speakers, out=out_dir), configs
        )
    else:  # one custom sentence
        make_results.TESTS[lang] = {"custom": text}
        make_results.main(
            SimpleNamespace(lang=lang, restore_step=restore_step, speakers=[speaker] if speaker else None, out=out_dir),
            configs,
        )
