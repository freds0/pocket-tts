"""Datasets and collation for pocket-tts training.

Two sources, one common sample format
`{"text": str, "waveform": FloatTensor[L], "speaker_id": str | None}` at the
model sample rate (24 kHz). `speaker_id` groups same-speaker utterances for
voice-prompt pairing in `PocketTTSTraining._sample_voice_prompt`; `None`
means "no reliable speaker info" and disables pairing for that sample.

- `LJSpeechDataset`: local LJSpeech-format corpora (e.g. BRSpeech-LN,
  `metadata.csv` with `id|text|normalized_text` plus a `wavs/` directory).
- `TagarelaStreaming`: `freds0/TAGARELA` streamed from the Hugging Face Hub —
  no bulk download; FLAC bytes are decoded with soundfile per sample.

Tokenization happens in the collate function (SentencePiece is instantiated
lazily per dataloader worker since the processor is not picklable).
"""

import hashlib
import io
import json
import logging
import random
import wave
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from pocket_tts.models.tts_model import prepare_text_prompt

logger = logging.getLogger(__name__)

MODEL_SAMPLE_RATE = 24_000

_resamplers: dict[int, torchaudio.transforms.Resample] = {}


def resample_to_model_rate(waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """Resample a [C, L] or [L] float waveform to 24 kHz mono [L]."""
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    if orig_sr != MODEL_SAMPLE_RATE:
        if orig_sr not in _resamplers:
            _resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, MODEL_SAMPLE_RATE)
        waveform = _resamplers[orig_sr](waveform)
    return waveform


def _stable_hash(key: str) -> float:
    """Deterministic uniform [0, 1) value for train/val splitting."""
    digest = hashlib.sha1(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _speaker_id(utt_id: str, root: Path) -> str:
    """Best-effort speaker key for same-speaker voice-prompt pairing.

    BRSpeech-LN ids are `{speaker}_{session}_...` (e.g. "2961_10229_...");
    LJSpeech ids ("LJ001-0001") have no such prefix, so every item collapses
    to one speaker key per corpus root (correct: LJSpeech is single-speaker).
    """
    prefix = utt_id.split("_", 1)[0] if "_" in utt_id else root.name
    return f"{root.name}:{prefix}"


class LJSpeechDataset(Dataset):
    """LJSpeech-format dataset (used for BRSpeech-LN pipeline validation)."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        val_fraction: float = 0.01,
        min_duration: float = 1.0,
        max_duration: float = 20.0,
        metadata_name: str = "metadata.csv",
        cache_dir: str | Path | None = None,
    ):
        assert split in ("train", "val")
        self.root = Path(root)
        self.wavs_dir = self.root / "wavs"
        durations = self._load_durations(cache_dir)

        self.items: list[tuple[str, str, str]] = []
        with open(self.root / metadata_name, encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\n").split("|")
                if len(fields) < 2:
                    continue
                utt_id, text = fields[0], fields[-1].strip()
                duration = durations.get(utt_id)
                if not text or duration is None:
                    continue
                if not (min_duration <= duration <= max_duration):
                    continue
                is_val = _stable_hash(utt_id) < val_fraction
                if (split == "val") == is_val:
                    self.items.append((utt_id, text, _speaker_id(utt_id, self.root)))
        logger.info("LJSpeechDataset(%s): %d items in split %s", self.root, len(self.items), split)

    def _load_durations(self, cache_dir: str | Path | None) -> dict[str, float]:
        cache_root = Path(cache_dir) if cache_dir else self.root
        cache_file = cache_root / "durations_cache.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        logger.info("Scanning wav durations in %s (first run only)...", self.wavs_dir)
        durations = {}
        for wav_path in self.wavs_dir.glob("*.wav"):
            try:
                with wave.open(str(wav_path)) as w:
                    durations[wav_path.stem] = w.getnframes() / w.getframerate()
            except (wave.Error, EOFError):
                logger.warning("Skipping unreadable wav: %s", wav_path)
        try:
            cache_file.write_text(json.dumps(durations))
        except OSError:
            logger.warning("Could not write duration cache to %s", cache_file)
        return durations

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        import soundfile

        utt_id, text, speaker_id = self.items[index]
        data, sr = soundfile.read(str(self.wavs_dir / f"{utt_id}.wav"), dtype="float32")
        waveform = torch.from_numpy(data.T if data.ndim == 2 else data)
        return {
            "text": text,
            "waveform": resample_to_model_rate(waveform, sr),
            "speaker_id": speaker_id,
        }


class TagarelaStreaming(IterableDataset):
    """Streams freds0/TAGARELA (16 kHz FLAC) from the HF Hub without downloading it.

    The first `val_take` examples of the (shuffled-by-shard) stream are reserved
    for validation; training skips them. Audio is decoded from raw FLAC bytes
    with soundfile, so no torchcodec/ffmpeg dependency is needed.
    """

    def __init__(
        self,
        split: str = "train",
        repo_id: str = "freds0/TAGARELA",
        val_take: int = 200,
        shuffle_buffer: int = 1000,
        min_duration: float = 1.0,
        max_duration: float = 20.0,
        seed: int = 42,
    ):
        assert split in ("train", "val")
        self.split = split
        self.repo_id = repo_id
        self.val_take = val_take
        self.shuffle_buffer = shuffle_buffer
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.seed = seed

    def _build_stream(self):
        import datasets as hf_datasets

        stream = hf_datasets.load_dataset(self.repo_id, split="train", streaming=True)
        stream = stream.cast_column("audio", hf_datasets.Audio(decode=False))
        if self.split == "val":
            return stream.take(self.val_take)
        stream = stream.skip(self.val_take)
        return stream.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

    def __iter__(self):
        import soundfile

        stream = self._build_stream()
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            # Shard the sample stream across dataloader workers.
            stream = _skip_take_interleave(stream, worker.id, worker.num_workers)
        for sample in stream:
            try:
                audio_bytes = sample["audio"]["bytes"]
                data, sr = soundfile.read(io.BytesIO(audio_bytes), dtype="float32")
            except (KeyError, RuntimeError, soundfile.LibsndfileError):
                continue
            duration = len(data) / sr
            if not (self.min_duration <= duration <= self.max_duration):
                continue
            text = (sample.get("sentence") or "").strip()
            if not text:
                continue
            waveform = torch.from_numpy(data.T if data.ndim == 2 else data)
            # No reliable per-speaker field in this corpus (podcast episodes,
            # not speaker-labeled): speaker_id=None disables voice-prompt
            # pairing for these samples rather than risk pairing across
            # speakers (see PocketTTSTraining._sample_voice_prompt).
            yield {
                "text": text,
                "waveform": resample_to_model_rate(waveform, sr),
                "speaker_id": None,
            }


def _skip_take_interleave(iterable, worker_id: int, num_workers: int):
    for i, item in enumerate(iterable):
        if i % num_workers == worker_id:
            yield item


class MixedStreaming(IterableDataset):
    """Probabilistic mix of a map-style dataset (cycled) and an iterable one."""

    def __init__(self, map_dataset: Dataset, iterable_dataset: IterableDataset, map_prob: float):
        self.map_dataset = map_dataset
        self.iterable_dataset = iterable_dataset
        self.map_prob = map_prob

    def __iter__(self):
        worker = get_worker_info()
        rng = random.Random(0 if worker is None else worker.seed)
        iterator = iter(self.iterable_dataset)
        while True:
            if len(self.map_dataset) > 0 and rng.random() < self.map_prob:
                yield self.map_dataset[rng.randrange(len(self.map_dataset))]
            else:
                try:
                    yield next(iterator)
                except StopIteration:
                    return


class Collate:
    """Tokenizes text and right-pads waveforms.

    Produces: waveforms [B, L], wave_lengths [B], text_tokens (list of [T_i] long).
    The SentencePiece processor is created lazily in each dataloader worker.
    """

    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        self._sp = None

    def __getstate__(self):
        return {"tokenizer_path": self.tokenizer_path, "_sp": None}

    def _tokenize(self, text: str) -> torch.Tensor:
        if self._sp is None:
            import sentencepiece

            from pocket_tts.utils.utils import download_if_necessary

            self._sp = sentencepiece.SentencePieceProcessor(
                str(download_if_necessary(self.tokenizer_path))
            )
        prepared, _ = prepare_text_prompt(
            text, pad_with_spaces_for_short_inputs=False, remove_semicolons=True
        )
        return torch.tensor(self._sp.encode(prepared, out_type=int), dtype=torch.long)

    def __call__(self, batch: list[dict]) -> dict:
        lengths = torch.tensor([item["waveform"].shape[0] for item in batch], dtype=torch.long)
        waveforms = torch.zeros(len(batch), int(lengths.max()))
        for i, item in enumerate(batch):
            waveforms[i, : item["waveform"].shape[0]] = item["waveform"]
        return {
            "waveforms": waveforms,
            "wave_lengths": lengths,
            "text_tokens": [self._tokenize(item["text"]) for item in batch],
            "speaker_ids": [item.get("speaker_id") for item in batch],
        }
