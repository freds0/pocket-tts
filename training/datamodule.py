"""LightningDataModule for pocket-tts training."""

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from training.data import Collate, LJSpeechDataset, MixedStreaming, TagarelaStreaming


class PocketTTSDataModule(pl.LightningDataModule):
    """Serves batches of raw waveforms + tokenized text.

    Args:
        source: "brspeech" (local LJSpeech format), "tagarela" (HF streaming),
            or "mix" (probabilistic mix, `brspeech_prob` from the local corpus).
        tokenizer_path: SentencePiece model path (local or hf:// URL); must match
            the FlowLM checkpoint being trained.
    """

    def __init__(
        self,
        source: str = "brspeech",
        tokenizer_path: str = (
            "hf://kyutai/pocket-tts-without-voice-cloning/languages/portuguese_24l/"
            "tokenizer.model@d29db7978e464fb90cb3359ee0c69a273b9142cc"
        ),
        brspeech_root: str = "/media/fred/FRED5TB/DATASETS/BRSpeech-LN",
        brspeech_prob: float = 0.2,
        batch_size: int = 4,
        num_workers: int = 4,
        min_duration: float = 1.0,
        max_duration: float = 20.0,
        val_fraction: float = 0.01,
        tagarela_val_take: int = 200,
        shuffle_buffer: int = 1000,
        seed: int = 42,
    ):
        super().__init__()
        assert source in ("brspeech", "tagarela", "mix")
        self.save_hyperparameters()
        self.collate = Collate(tokenizer_path)

    def _brspeech(self, split: str) -> LJSpeechDataset:
        p = self.hparams
        return LJSpeechDataset(
            p.brspeech_root,
            split=split,
            val_fraction=p.val_fraction,
            min_duration=p.min_duration,
            max_duration=p.max_duration,
        )

    def _tagarela(self, split: str) -> TagarelaStreaming:
        p = self.hparams
        return TagarelaStreaming(
            split=split,
            val_take=p.tagarela_val_take,
            shuffle_buffer=p.shuffle_buffer,
            min_duration=p.min_duration,
            max_duration=p.max_duration,
            seed=p.seed,
        )

    def setup(self, stage: str | None = None):
        source = self.hparams.source
        if source == "brspeech":
            self.train_dataset = self._brspeech("train")
            self.val_dataset = self._brspeech("val")
        elif source == "tagarela":
            self.train_dataset = self._tagarela("train")
            self.val_dataset = self._tagarela("val")
        else:
            self.train_dataset = MixedStreaming(
                self._brspeech("train"), self._tagarela("train"), self.hparams.brspeech_prob
            )
            # Validate on both distributions: local val is cheap and stable.
            self.val_dataset = self._brspeech("val")

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        is_iterable = not hasattr(dataset, "__len__")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle and not is_iterable,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=not is_iterable,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)
