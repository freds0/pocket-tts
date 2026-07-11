"""Training entry point.

    python -m training.train fit --config training/configs/brspeech_validate.yaml

Uses LightningCLI: any hyperparameter of PocketTTSTraining / PocketTTSDataModule
and any Trainer flag is overridable from the YAML or the command line.
"""

import os

# Keep HF downloads (weights, tokenizer, TAGARELA cache) out of ~/.cache by
# default; relative to the cwd training.train is run from, like ./logs/.
os.environ.setdefault("HF_HOME", "./cache/hf_cache")

import torch  # noqa: E402
from lightning.pytorch.cli import LightningCLI  # noqa: E402

from training.datamodule import PocketTTSDataModule  # noqa: E402
from training.lightning_module import PocketTTSTraining  # noqa: E402


def main():
    # pocket_tts.models.tts_model sets torch.set_num_threads(1) at import time
    # (a CPU-inference optimization); undo it for training.
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    torch.set_float32_matmul_precision("high")
    LightningCLI(PocketTTSTraining, PocketTTSDataModule)


if __name__ == "__main__":
    main()
