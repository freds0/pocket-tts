"""Teacher-forced sequence assembly for pocket-tts training.

Replicates the exact inference-time layout of `TTSModel` (see
`get_state_for_audio_prompt` and `_run_flow_lm`):

    [ bos_before_voice? | speaker-projected voice latents | text embeddings |
      BOS + normalized latents[:-1] ]

The transformer output at each position of the last block conditions the flow
head to predict the corresponding normalized latent frame; the same outputs
feed the EOS head. With `model_state=None`, `StreamingTransformer` applies a
full-sequence causal mask, so batches are right-padded and padded positions are
simply masked out of the loss.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.models.flow_lm import FlowLMModel


@dataclass
class AssembledBatch:
    embeddings: torch.Tensor  # [B, L, D] transformer input embeddings, right-padded
    cond_positions: torch.Tensor  # [B, S] long, positions whose outputs predict frames
    target_latents: torch.Tensor  # [B, S, ldim] normalized latents (data at time 1)
    frame_mask: torch.Tensor  # [B, S] bool, True on valid frames
    eos_labels: torch.Tensor  # [B, S] float, 1.0 on the final valid frame


def gather_conditioning(
    flow_lm: FlowLMModel, transformer_out: torch.Tensor, batch: AssembledBatch
) -> torch.Tensor:
    """Select (and layer-norm) the outputs that condition frame predictions.

    Equivalent to `FlowLMModel.backbone`'s `out_norm` + slicing, generalized to
    per-sample offsets. Returns [B, S, D] float32.
    """
    index = batch.cond_positions[..., None].expand(-1, -1, transformer_out.shape[-1])
    cond = transformer_out.gather(1, index)
    return flow_lm.out_norm(cond.to(torch.float32))


def assemble_batch(
    flow_lm: FlowLMModel,
    text_tokens: list[torch.Tensor],
    voice_latents: list[torch.Tensor | None],
    target_latents: list[torch.Tensor],
) -> AssembledBatch:
    """Build the padded teacher-forcing batch.

    Args:
        flow_lm: the model (used for its embedding tables; runs under autograd).
        text_tokens: per sample, long tensor [T_text] (may be empty for CFG dropout).
        voice_latents: per sample, *unnormalized* Mimi latents [T_voice, ldim]
            for the voice prompt, or None (matches `TTSModel._encode_audio`,
            which projects raw latents with `speaker_proj_weight`).
        target_latents: per sample, *normalized* latents [S, ldim]
            (i.e. (mimi_latent - emb_mean) / emb_std), S >= 1.
    """
    assert len(text_tokens) == len(voice_latents) == len(target_latents)
    device = flow_lm.bos_emb.device
    dtype = flow_lm.bos_emb.dtype

    per_sample_embeddings: list[torch.Tensor] = []
    offsets: list[int] = []
    for tokens, voice, target in zip(text_tokens, voice_latents, target_latents):
        parts = []
        if voice is not None and voice.shape[0] > 0:
            projected = F.linear(voice.to(dtype), flow_lm.speaker_proj_weight)
            if flow_lm.insert_bos_before_voice:
                parts.append(flow_lm.bos_before_voice[0])
            parts.append(projected)
        if tokens.numel() > 0:
            parts.append(flow_lm.conditioner(TokenizedText(tokens.to(device)[None]))[0])
        audio_in = torch.cat([flow_lm.bos_emb[None], target.to(dtype)[:-1]], dim=0)
        parts.append(flow_lm.input_linear(audio_in))
        emb = torch.cat(parts, dim=0)
        per_sample_embeddings.append(emb)
        offsets.append(emb.shape[0] - audio_in.shape[0])

    batch_size = len(per_sample_embeddings)
    max_len = max(e.shape[0] for e in per_sample_embeddings)
    max_frames = max(t.shape[0] for t in target_latents)
    ldim = target_latents[0].shape[-1]

    embeddings = torch.zeros(batch_size, max_len, flow_lm.dim, device=device, dtype=dtype)
    cond_positions = torch.zeros(batch_size, max_frames, device=device, dtype=torch.long)
    targets = torch.zeros(batch_size, max_frames, ldim, device=device, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_frames, device=device, dtype=torch.bool)
    eos_labels = torch.zeros(batch_size, max_frames, device=device, dtype=torch.float32)

    for i, (emb, target, offset) in enumerate(
        zip(per_sample_embeddings, target_latents, offsets)
    ):
        n_frames = target.shape[0]
        embeddings[i, : emb.shape[0]] = emb
        cond_positions[i, :n_frames] = offset + torch.arange(n_frames, device=device)
        targets[i, :n_frames] = target.to(torch.float32)
        frame_mask[i, :n_frames] = True
        eos_labels[i, n_frames - 1] = 1.0

    return AssembledBatch(embeddings, cond_positions, targets, frame_mask, eos_labels)
