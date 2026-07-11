"""LightningModule for fine-tuning (or retraining) the pocket-tts FlowLM.

Training recipe from arXiv:2509.06926 (Table 14, Text-to-Speech column):
AdamW(0.9, 0.95), cosine schedule, head batch multiplier 8, flow-matching loss
on 75% of head samples + LSD loss on 25% (Appendix A), EOS head, text-conditioning
dropout to enable latent CFG at inference. The Mimi VAE is frozen and used only
to extract target latents; `emb_mean`/`emb_std` statistics are kept from the
checkpoint.
"""

import contextlib
import copy
import logging
import math
import random

import lightning.pytorch as pl
import torch
from torch import nn

from pocket_tts.default_parameters import DEFAULT_EOS_THRESHOLD
from pocket_tts.models.tts_model import TTSModel
from training.losses import AdaptiveWeight, eos_loss, flow_matching_loss, lsd_loss
from training.sequence import assemble_batch, gather_conditioning

logger = logging.getLogger(__name__)


class PocketTTSTraining(pl.LightningModule):
    def __init__(
        self,
        model_config: str = "portuguese_24l",
        pretrained: bool = True,
        lr: float = 3e-5,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 500,
        lr_min_ratio: float = 0.1,
        head_batch_multiplier: int = 8,
        fm_fraction: float = 0.75,
        adaptive_weighting: bool = True,
        eos_weight: float = 1.0,
        eos_pos_weight: float = 1.0,
        text_dropout: float = 0.15,
        voice_prompt_prob: float = 0.5,
        voice_prompt_min_frames: int = 13,  # ~1 s at 12.5 Hz
        voice_prompt_max_frames: int = 50,  # ~4 s
        activation_checkpointing: bool = False,
        val_sample_texts: list[str] | None = None,
        val_voice_prompt: str | None = None,
        synthesize_every_n_vals: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_epoch_count = 0
        self.adaptive_fm = AdaptiveWeight() if adaptive_weighting else None
        self.adaptive_lsd = AdaptiveWeight() if adaptive_weighting else None
        self.flow_lm: nn.Module | None = None
        self.mimi: nn.Module | None = None

    # ---------------------------------------------------------------- setup

    def configure_model(self):
        if self.flow_lm is not None:
            return
        tts = TTSModel.load_model(language=self.hparams.model_config)
        self._config = tts.config
        self.flow_lm = tts.flow_lm
        self.mimi = tts.mimi
        if not self.hparams.pretrained:
            logger.warning("Reinitializing FlowLM weights (training from scratch). "
                           "Mimi VAE and latent statistics are kept from the checkpoint.")
            emb_mean, emb_std = self.flow_lm.emb_mean.clone(), self.flow_lm.emb_std.clone()
            self.flow_lm.apply(_reset_parameters)
            self.flow_lm.emb_mean.copy_(emb_mean)
            self.flow_lm.emb_std.copy_(emb_std)
        self.mimi.requires_grad_(False)
        self.mimi.eval()
        encoder_ok = any(
            p.abs().sum() > 0 for p in self.mimi.encoder.parameters()
        )
        if not encoder_ok:
            raise RuntimeError(
                "The Mimi encoder weights are all zeros: you are using the public "
                "'without-voice-cloning' checkpoint, which cannot encode audio into "
                "training latents. Accept the terms at "
                "https://huggingface.co/kyutai/pocket-tts and run `hf auth login`."
            )
        if self.hparams.eos_weight == 0:
            # Actually freeze the head: 0 * loss still leaves it in the
            # optimizer, where AdamW weight decay erodes its calibration.
            self.flow_lm.out_eos.requires_grad_(False)
        if self.hparams.activation_checkpointing:
            _apply_activation_checkpointing(self.flow_lm.transformer)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mimi is not None:
            self.mimi.eval()  # frozen VAE stays in eval mode
        return self

    @property
    def samples_per_frame(self) -> int:
        return int(self._config.mimi.sample_rate / self._config.mimi.frame_rate)

    # ---------------------------------------------------------------- steps

    def _encode_latents(self, waveforms: torch.Tensor) -> torch.Tensor:
        """[B, L] float32 audio -> [B, T, ldim] raw (unnormalized) latents."""
        with torch.no_grad(), torch.autocast(device_type=waveforms.device.type, enabled=False):
            latents = self.mimi.encode_to_latent(waveforms[:, None, :].float())
        return latents.transpose(1, 2).to(torch.float32)

    def _sample_voice_prompt(
        self,
        raw_latents: torch.Tensor,
        frame_counts: list[int],
        speaker_ids: list | None,
        i: int,
        rng: random.Random,
    ) -> torch.Tensor | None:
        """Pick a voice-prompt snippet from a DIFFERENT utterance of the same
        speaker in this batch — matches inference (an unrelated reference
        clip conditions generation of the full target text).

        The previous approach cut the prompt from the target utterance
        itself while keeping the full transcript as text conditioning, which
        taught the model that some prefix of the text was already "spoken"
        by the prompt — a mismatch that doesn't exist at inference (the
        prompt there is unrelated audio) and that produced garbled/skipped
        words. Falls back to no prompt (not the old self-cut) when no
        same-speaker candidate is available in the batch (e.g. TAGARELA,
        which has no reliable speaker field, or small multi-speaker batches).
        """
        p = self.hparams
        if speaker_ids is None or speaker_ids[i] is None:
            return None
        if rng.random() >= p.voice_prompt_prob:
            return None
        candidates = [
            j
            for j, sid in enumerate(speaker_ids)
            if j != i and sid == speaker_ids[i] and frame_counts[j] >= p.voice_prompt_min_frames
        ]
        if not candidates:
            return None
        j = rng.choice(candidates)
        n_j = frame_counts[j]
        k = rng.randint(p.voice_prompt_min_frames, min(p.voice_prompt_max_frames, n_j))
        start = rng.randint(0, n_j - k)
        return raw_latents[j, start : start + k]

    def _step(self, batch: dict, batch_idx: int, stage: str) -> torch.Tensor:
        p = self.hparams
        deterministic = stage == "val"
        rng = random.Random(batch_idx if deterministic else None)
        generator = None
        if deterministic:
            generator = torch.Generator(device=self.device).manual_seed(batch_idx)

        raw_latents = self._encode_latents(batch["waveforms"])
        frame_counts = [
            min(math.ceil(int(n) / self.samples_per_frame), raw_latents.shape[1])
            for n in batch["wave_lengths"]
        ]

        speaker_ids = batch.get("speaker_ids")
        text_tokens, voice_latents, target_latents = [], [], []
        emb_mean, emb_std = self.flow_lm.emb_mean, self.flow_lm.emb_std
        for i, tokens in enumerate(batch["text_tokens"]):
            target_raw = raw_latents[i, : frame_counts[i]]
            voice = self._sample_voice_prompt(raw_latents, frame_counts, speaker_ids, i, rng)
            if rng.random() < p.text_dropout and stage == "train":
                tokens = tokens[:0]
            text_tokens.append(tokens)
            voice_latents.append(voice)
            target_latents.append((target_raw - emb_mean) / emb_std)

        assembled = assemble_batch(self.flow_lm, text_tokens, voice_latents, target_latents)
        transformer_out = self.flow_lm.transformer(assembled.embeddings, None)

        with torch.autocast(device_type=self.device.type, enabled=False):
            cond = gather_conditioning(self.flow_lm, transformer_out, assembled)

            # With eos_weight == 0 the head is frozen and only monitored.
            with torch.no_grad() if p.eos_weight == 0 else contextlib.nullcontext():
                eos_logits = self.flow_lm.out_eos(cond)[..., 0]
                loss_eos = eos_loss(
                    eos_logits, assembled.eos_labels, assembled.frame_mask, p.eos_pos_weight
                )

            cond_flat = cond[assembled.frame_mask]
            x1_flat = assembled.target_latents[assembled.frame_mask]
            # Head batch multiplier: reuse each backbone output for several
            # independent (t, eps) draws (paper Sec. 4.4).
            cond_rep = cond_flat.repeat(p.head_batch_multiplier, 1)
            x1_rep = x1_flat.repeat(p.head_batch_multiplier, 1)
            n_fm = max(1, int(p.fm_fraction * cond_rep.shape[0]))
            loss_fm, aux_fm = flow_matching_loss(
                self.flow_lm.flow_net,
                cond_rep[:n_fm],
                x1_rep[:n_fm],
                self.adaptive_fm,
                generator,
                return_aux=True,
            )
            if n_fm < cond_rep.shape[0]:
                loss_lsd, aux_lsd = lsd_loss(
                    self.flow_lm.flow_net,
                    cond_rep[n_fm:],
                    x1_rep[n_fm:],
                    self.adaptive_lsd,
                    generator,
                    return_aux=True,
                )
            else:
                loss_lsd = torch.zeros_like(loss_fm)
                aux_lsd = {k: torch.zeros_like(v) for k, v in aux_fm.items()}

        loss = loss_fm + loss_lsd
        if p.eos_weight > 0:
            loss = loss + p.eos_weight * loss_eos
        frames = int(assembled.frame_mask.sum())

        # EOS calibration at the deployed threshold: mean logit on the true EOS
        # frame, and how often earlier frames would (wrongly) fire.
        with torch.no_grad():
            is_eos = assembled.eos_labels.bool()
            non_eos = assembled.frame_mask & ~is_eos
            eos_logit_true = eos_logits[is_eos].mean()
            eos_false_trigger = (
                (eos_logits[non_eos] > DEFAULT_EOS_THRESHOLD).float().mean()
                if int(non_eos.sum()) > 0
                else torch.zeros((), device=self.device)
            )

        batch_size = len(batch["text_tokens"])
        self.log_dict(
            {
                f"{stage}/loss_fm": loss_fm,
                f"{stage}/loss_lsd": loss_lsd,
                f"{stage}/loss_eos": loss_eos,
                f"{stage}/loss_total": loss,
            },
            prog_bar=stage == "train",
            batch_size=batch_size,
            sync_dist=stage == "val",
        )
        self.log_dict(
            {
                # Raw errors: the adaptive-weighted losses above can fall from
                # w drift alone; these are the signals to trust (and monitor).
                f"{stage}/mse_fm": aux_fm["mse"],
                f"{stage}/mse_lsd": aux_lsd["mse"],
                f"{stage}/w_fm": aux_fm["w"],
                f"{stage}/w_lsd": aux_lsd["w"],
                f"{stage}/eos_logit_true": eos_logit_true,
                f"{stage}/eos_false_trigger": eos_false_trigger,
                f"{stage}/frames_per_batch": float(frames),
            },
            batch_size=batch_size,
            sync_dist=stage == "val",
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def on_before_optimizer_step(self, optimizer):
        # Pre-clip gradient norm (norm-of-norms avoids materializing a flat copy).
        norms = [p.grad.norm(2) for p in self.parameters() if p.grad is not None]
        if norms:
            self.log("train/grad_norm", torch.stack(norms).norm(2))

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "val")

    # ------------------------------------------------------------ synthesis

    def on_validation_epoch_end(self):
        texts = self.hparams.val_sample_texts
        if not texts or self.hparams.val_voice_prompt is None or self.trainer.sanity_checking:
            return
        # CPU synthesis of the 24-layer model takes minutes and stalls the GPU;
        # only do it every N validation epochs.
        self._val_epoch_count += 1
        if (self._val_epoch_count - 1) % self.hparams.synthesize_every_n_vals != 0:
            return
        try:
            self._synthesize_and_log(texts)
        except Exception:
            logger.exception("Validation-time synthesis failed (training continues).")

    @torch.no_grad()
    def _synthesize_and_log(self, texts: list[str]):
        from pathlib import Path

        config = copy.deepcopy(self._config)
        config.weights_path = None
        config.flow_lm.weights_path = None
        config.mimi.weights_path = None
        # Deployed threshold: with anything laxer, the TensorBoard audio hides
        # the early-truncation failures the exported model would actually show.
        tts = TTSModel._from_pydantic_config_with_weights(
            config,
            temp=0.7,
            lsd_decode_steps=1,
            noise_clamp=None,
            eos_threshold=DEFAULT_EOS_THRESHOLD,
        )
        tts.flow_lm.load_state_dict(
            {k: v.detach().cpu().float() for k, v in self.flow_lm.state_dict().items()}
        )
        tts.mimi.load_state_dict(
            {k: v.detach().cpu().float() for k, v in self.mimi.state_dict().items()}
        )
        tts.eval()
        voice_state = tts.get_state_for_audio_prompt(Path(self.hparams.val_voice_prompt))
        for i, text in enumerate(texts):
            audio = tts.generate_audio(voice_state, text, copy_state=True)
            self._log_audio(f"val_samples/{i}", audio, tts.sample_rate)

    def _log_audio(self, key: str, audio: torch.Tensor, sample_rate: int):
        # self.logger is only trainer.loggers[0]; with wandb + TensorBoard both
        # configured, iterate explicitly so every logger gets the sample.
        for lg in self.loggers:
            if isinstance(lg, pl.loggers.TensorBoardLogger):
                lg.experiment.add_audio(
                    key, audio.reshape(-1, 1).numpy(), global_step=self.global_step,
                    sample_rate=sample_rate,
                )
            elif isinstance(lg, pl.loggers.WandbLogger):
                lg.log_audio(
                    key=key, audios=[audio.numpy()], sample_rate=[sample_rate],
                    step=self.global_step,
                )

    # ------------------------------------------------------------ optimizer

    def configure_optimizers(self):
        p = self.hparams
        params = [q for q in self.parameters() if q.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=p.lr, betas=tuple(p.betas), weight_decay=p.weight_decay
        )
        max_steps = self.trainer.max_steps if self.trainer.max_steps > 0 else 100_000

        def lr_lambda(step: int) -> float:
            if step < p.warmup_steps:
                return step / max(1, p.warmup_steps)
            progress = (step - p.warmup_steps) / max(1, max_steps - p.warmup_steps)
            progress = min(1.0, progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return p.lr_min_ratio + (1.0 - p.lr_min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def _reset_parameters(module: nn.Module):
    if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
        module.reset_parameters()


def _apply_activation_checkpointing(transformer: nn.Module):
    from torch.utils.checkpoint import checkpoint

    for layer in transformer.layers:
        original_forward = layer.forward

        def wrapped(x, model_state, _orig=original_forward):
            if model_state is None and torch.is_grad_enabled():
                return checkpoint(_orig, x, None, use_reentrant=False)
            return _orig(x, model_state)

        layer.forward = wrapped
