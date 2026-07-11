# pocket-tts training pipeline (PyTorch Lightning)

Fine-tunes (or retrains) the pocket-tts CALM model — recipe from
[arXiv:2509.06926](https://arxiv.org/abs/2509.06926) (Table 14 TTS column +
Appendix A LSD losses). Default target: the 24-layer Portuguese teacher
(`portuguese_24l`, ~313M).

## One-time setup

1. Conda env: `pocket-tts` with the base package plus the training extras —
   ```bash
   conda run -n pocket-tts pip install -e ".[audio]"
   conda run -n pocket-tts pip install -r training/requirements.txt
   ```
2. **Gated weights (required for training).** The public
   `kyutai/pocket-tts-without-voice-cloning` checkpoints ship a **zeroed Mimi
   encoder**, so they cannot produce training latents. Accept the terms at
   <https://huggingface.co/kyutai/pocket-tts>, then:
   ```bash
   conda run -n pocket-tts hf auth login
   ```
3. HF caches (weights, tokenizer, TAGARELA) and Lightning logs/checkpoints
   all default to `./cache/` and `./logs/`, relative to the cwd
   `training.train` is run from (`train.py` sets `HF_HOME` automatically).
   To use a different disk (e.g. a nearly-full `/home`), export `HF_HOME`
   yourself before running, or edit `trainer.logger[*].init_args.save_dir` /
   `callbacks[*].init_args.dirpath` in the config.

## Running

```bash
conda activate pocket-tts

# Unit tests (no network / weights needed)
pytest training/tests -v

# Pipeline validation on local BRSpeech-LN (LJSpeech format)
python -m training.train fit --config training/configs/brspeech_validate.yaml

# Streaming fine-tune on freds0/TAGARELA (no bulk download) mixed with BRSpeech
python -m training.train fit --config training/configs/tagarela_stream.yaml

# Fine-tune on the standard English LJSpeech-1.1 dataset (see "LJSpeech (English)" below)
python -m training.train fit --config training/configs/ljspeech_english.yaml

# Resume
python -m training.train fit --config <cfg> --ckpt_path <...>/last.ckpt

# Export for inference with the stock CLI
python -m training.export <checkpoint>.ckpt ./logs/export/run1
pocket-tts generate --config .../export/run1/config.yaml --text "Olá mundo" --output out.wav

# Monitor (logs/checkpoints are written under ./logs/, relative to the cwd
# training.train was run from)
tensorboard --logdir ./logs/tensorboard
```

## LJSpeech (English)

[LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) is a public single-speaker
English corpus (~24h, 22.05 kHz), already in the `metadata.csv` + `wavs/` layout
`LJSpeechDataset` expects — no code changes needed, just point `data.brspeech_root`
at it (the loader is generic; the option is named after its first use case,
BRSpeech-LN).

```bash
cd /media/fred/FRED5TB/DATASETS
curl -L -o LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjf LJSpeech-1.1.tar.bz2   # -> LJSpeech-1.1/{metadata.csv,wavs/}

conda activate pocket-tts
python -m training.train fit --config training/configs/ljspeech_english.yaml
```

This targets `english` (the 6-layer, ~90M shipped model) — English has no
24-layer teacher checkpoint like Portuguese/French/German/Italian/Spanish do.

## What is trained

- **Trained:** FlowLM backbone transformer, LSD flow head (`SimpleMLPAdaLN`),
  EOS head (only when `eos_weight > 0`; at `0.0` it is truly frozen —
  `requires_grad_(False)` — to preserve the pretrained calibration), text
  embedding table, `speaker_proj_weight`, BOS embeddings, and a learned
  adaptive loss weighting `w_psi(s, t)` (training-only, not exported).
- **Frozen:** the entire Mimi VAE (target-latent extractor + vocoder) and the
  latent normalization statistics `emb_mean`/`emb_std`.

Sequence layout replicates inference exactly
(`[bos_before_voice | speaker-projected voice prompt | text | BOS | latents]`);
voice prompts are a random 1–4 s latent snippet from a *different* utterance
of the same speaker within the batch (matching inference: an unrelated
reference clip conditions generation of the full target text), and text
conditioning is dropped with p=0.15 so latent CFG (alpha≈1.5) keeps working
at inference. Speaker pairing needs a `speaker_id` per sample (derived from
the BRSpeech-LN/LJSpeech filename convention); TAGARELA has no reliable
speaker field, so its samples train without a voice prompt.

## Notes / caveats

- TAGARELA is 16 kHz (band-limited after resampling to 24 kHz) and licensed
  **CC BY-NC-SA** — fine-tuned weights inherit non-commercial terms. The mix
  config keeps 20% BRSpeech (22.05 kHz) to preserve high-band energy.
- 12 GB VRAM: micro-batch 4 + grad accumulation. If OOM, set
  `model.activation_checkpointing: true` or reduce `data.max_duration`.
- From-scratch training (`model.pretrained: false`) reinitializes the FlowLM
  but keeps the VAE and latent stats; the paper recipe used 8xH100 for 400k
  steps at batch 128x60 s, so from scratch is only realistic on a cluster.
