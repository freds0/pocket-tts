"""Export a Lightning checkpoint to inference-ready pocket-tts artifacts.

Produces `model.safetensors` (keys `flow_lm.*` / `mimi.*`, exactly the
`TTSModel.state_dict()` layout loaded by `config.weights_path`) plus a YAML
config so the result works with the stock CLI:

    pocket-tts generate --config <out_dir>/config.yaml --text "Olá mundo"

Usage:
    python -m training.export <lightning.ckpt> <out_dir> [--base-config portuguese_24l]
"""

import argparse
from pathlib import Path

import safetensors.torch
import torch
import yaml

from pocket_tts.models.tts_model import CONFIGS_DIR


def export_checkpoint(ckpt_path: str | Path, out_dir: str | Path, base_config: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    exported = {
        key: tensor.float().contiguous()
        for key, tensor in state_dict.items()
        if key.startswith(("flow_lm.", "mimi."))
    }
    if not exported:
        raise ValueError(f"No flow_lm.*/mimi.* keys found in {ckpt_path}")

    weights_file = out_dir / "model.safetensors"
    safetensors.torch.save_file(exported, str(weights_file))

    base_yaml = CONFIGS_DIR / f"{base_config}.yaml"
    config = yaml.safe_load(base_yaml.read_text())
    config["weights_path"] = str(weights_file.resolve())
    config["weights_path_without_voice_cloning"] = str(weights_file.resolve())
    config_file = out_dir / "config.yaml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpt", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--base-config", default="portuguese_24l")
    args = parser.parse_args()
    config_file = export_checkpoint(args.ckpt, args.out_dir, args.base_config)
    print(f"Exported. Test with:\n  pocket-tts generate --config {config_file} --text 'Olá mundo'")


if __name__ == "__main__":
    main()
