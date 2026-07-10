"""Unit tests for the training pipeline (no network, no real weights needed).

Run with: conda run -n pocket-tts pytest training/tests -v
"""

import wave
from pathlib import Path

import pytest
import sentencepiece as spm
import torch
from torch import nn
from torch.nn import functional as F

from pocket_tts.conditioners.text import LUTConditioner
from pocket_tts.models.flow_lm import FlowLMModel, lsd_decode
from pocket_tts.modules.mimi_transformer import StreamingTransformer
from pocket_tts.modules.mlp import SimpleMLPAdaLN
from training.losses import AdaptiveWeight, eos_loss, flow_matching_loss, lsd_loss
from training.sequence import assemble_batch, gather_conditioning

LDIM = 8
DIM = 64
VOCAB = 24


@pytest.fixture(scope="session")
def tokenizer_path(tmp_path_factory) -> str:
    """Train a tiny throwaway SentencePiece model."""
    tmp = tmp_path_factory.mktemp("spm")
    sentences = ["o rato roeu a roupa", "do rei de roma", "bom dia", "ola mundo"] * 30
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_prefix=str(tmp / "tok"),
        vocab_size=VOCAB,
        character_coverage=1.0,
    )
    return str(tmp / "tok.model")


@pytest.fixture(scope="session")
def flow_lm(tokenizer_path) -> FlowLMModel:
    torch.manual_seed(0)
    conditioner = LUTConditioner(
        n_bins=VOCAB, tokenizer_path=tokenizer_path, dim=DIM, output_dim=DIM
    )
    flow_net = SimpleMLPAdaLN(
        in_channels=LDIM,
        model_channels=32,
        out_channels=LDIM,
        cond_channels=DIM,
        num_res_blocks=2,
        num_time_conds=2,
    )
    transformer = StreamingTransformer(d_model=DIM, num_heads=4, num_layers=2)
    model = FlowLMModel(
        conditioner=conditioner,
        flow_net=flow_net,
        transformer=transformer,
        dim=DIM,
        ldim=LDIM,
        dtype=torch.float32,
        insert_bos_before_voice=True,
    )
    model.speaker_proj_weight = nn.Parameter(torch.randn(DIM, LDIM) * 0.02)
    return model


def test_adaptive_weight_starts_at_zero():
    w = AdaptiveWeight()
    s, t = torch.rand(5, 1), torch.rand(5, 1)
    assert w(s, t).shape == (5, 1)
    assert torch.allclose(w(s, t), torch.zeros(5, 1))


def test_flow_matching_loss_overfits(flow_lm):
    torch.manual_seed(0)
    flow_net = SimpleMLPAdaLN(LDIM, 64, LDIM, DIM, 3, num_time_conds=2)
    cond = torch.randn(16, DIM)
    x1 = torch.randn(16, LDIM)
    opt = torch.optim.Adam(flow_net.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(0)
    first = None
    for _ in range(300):
        opt.zero_grad()
        loss = flow_matching_loss(flow_net, cond, x1, generator=generator)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.5 * first, f"FM loss did not decrease: {first} -> {loss.item()}"


def test_flow_matching_then_sampling_recovers_target():
    """End-to-end sanity: train the head on a single (cond, x1), then 8-step
    lsd_decode from noise should land near x1 (verifies the loss matches the
    inference-time time convention)."""
    torch.manual_seed(0)
    flow_net = SimpleMLPAdaLN(LDIM, 64, LDIM, DIM, 3, num_time_conds=2)
    cond = torch.randn(1, DIM)
    x1 = torch.randn(1, LDIM)
    opt = torch.optim.Adam(flow_net.parameters(), lr=2e-3)
    for _ in range(800):
        opt.zero_grad()
        loss = flow_matching_loss(flow_net, cond.expand(64, -1), x1.expand(64, -1))
        loss.backward()
        opt.step()
    with torch.no_grad():
        from functools import partial

        noise = torch.randn(256, LDIM)
        recon = lsd_decode(partial(flow_net, cond.expand(256, -1)), noise, num_steps=8)
    err = (recon.mean(0) - x1[0]).abs().mean().item()
    assert err < 0.35, f"8-step decode mean {err} too far from target"


def test_lsd_loss_backward(flow_lm):
    torch.manual_seed(0)
    flow_net = SimpleMLPAdaLN(LDIM, 32, LDIM, DIM, 2, num_time_conds=2)
    weight_fn = AdaptiveWeight()
    cond = torch.randn(8, DIM, requires_grad=False)
    x1 = torch.randn(8, LDIM)
    loss = lsd_loss(flow_net, cond, x1, weight_fn)
    loss.backward()
    grads = [p.grad for p in flow_net.parameters() if p.grad is not None]
    assert grads, "no gradients reached the flow net through the jvp"
    assert all(torch.isfinite(g).all() for g in grads)


def test_eos_loss_masked():
    logits = torch.zeros(2, 5)
    labels = torch.zeros(2, 5)
    labels[0, 4] = 1.0
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 3:] = False
    loss = eos_loss(logits, labels, mask)
    assert torch.isfinite(loss) and loss.item() > 0


def _fake_batch(flow_lm):
    tokens_a = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    tokens_b = torch.tensor([9, 10], dtype=torch.long)
    voice_a = torch.randn(4, LDIM)  # sample A has a voice prompt
    target_a = torch.randn(6, LDIM)
    target_b = torch.randn(9, LDIM)
    return [tokens_a, tokens_b], [voice_a, None], [target_a, target_b]


def test_assemble_batch_layout(flow_lm):
    text, voice, targets = _fake_batch(flow_lm)
    batch = assemble_batch(flow_lm, text, voice, targets)

    # Sample A: 1 (bos_before_voice) + 4 (voice) + 4 (text) + 6 (audio) = 15
    # Sample B: 2 (text) + 9 (audio) = 11
    assert batch.embeddings.shape == (2, 15, DIM)
    assert batch.cond_positions[0, 0] == 9 and batch.cond_positions[1, 0] == 2
    assert batch.frame_mask.sum(dim=1).tolist() == [6, 9]
    assert batch.eos_labels[0, 5] == 1.0 and batch.eos_labels[1, 8] == 1.0
    assert batch.eos_labels.sum() == 2.0

    # First audio-input position must hold the embedded BOS latent.
    bos_embedding = flow_lm.input_linear(flow_lm.bos_emb[None])[0]
    assert torch.allclose(batch.embeddings[0, 9], bos_embedding, atol=1e-6)
    assert torch.allclose(batch.embeddings[1, 2], bos_embedding, atol=1e-6)
    # Voice prompt block of sample A starts with bos_before_voice.
    assert torch.allclose(batch.embeddings[0, 0], flow_lm.bos_before_voice[0, 0], atol=1e-6)
    # Padding stays zero.
    assert batch.embeddings[1, 11:].abs().sum() == 0


def test_full_teacher_forced_step(flow_lm):
    """Losses computed through the real transformer backward cleanly."""
    text, voice, targets = _fake_batch(flow_lm)
    batch = assemble_batch(flow_lm, text, voice, targets)
    out = flow_lm.transformer(batch.embeddings, None)
    cond = gather_conditioning(flow_lm, out, batch)
    assert cond.shape == (2, 9, DIM)

    cond_flat = cond[batch.frame_mask]
    x1_flat = batch.target_latents[batch.frame_mask]
    loss = (
        flow_matching_loss(flow_lm.flow_net, cond_flat, x1_flat)
        + lsd_loss(flow_lm.flow_net, cond_flat, x1_flat)
        + eos_loss(flow_lm.out_eos(cond)[..., 0], batch.eos_labels, batch.frame_mask)
    )
    loss.backward()
    named_missing = [
        name
        for name, p in flow_lm.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    # bos_before_voice only receives grad when a voice prompt is present (it is here).
    assert not named_missing, f"missing/nonfinite grads: {named_missing}"


def test_ljspeech_dataset(tmp_path):
    root = tmp_path / "corpus"
    (root / "wavs").mkdir(parents=True)
    for i, dur in enumerate([2.0, 0.2]):  # second file filtered out (< min_duration)
        with wave.open(str(root / "wavs" / f"utt{i}.wav"), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(b"\x00\x00" * int(22050 * dur))
    (root / "metadata.csv").write_text(
        "utt0|texto qualquer|texto qualquer\nutt1|curto|curto\n", encoding="utf-8"
    )

    from training.data import MODEL_SAMPLE_RATE, LJSpeechDataset

    dataset = LJSpeechDataset(root, split="train", val_fraction=0.0, min_duration=1.0)
    assert len(dataset) == 1
    item = dataset[0]
    assert item["text"] == "texto qualquer"
    assert item["waveform"].dim() == 1
    assert abs(item["waveform"].shape[0] - 2.0 * MODEL_SAMPLE_RATE) < 100


def test_collate(tokenizer_path):
    from training.data import Collate

    collate = Collate(tokenizer_path)
    batch = collate(
        [
            {"text": "o rato roeu", "waveform": torch.randn(1000)},
            {"text": "bom dia", "waveform": torch.randn(500)},
        ]
    )
    assert batch["waveforms"].shape == (2, 1000)
    assert batch["wave_lengths"].tolist() == [1000, 500]
    assert batch["waveforms"][1, 500:].abs().sum() == 0
    assert len(batch["text_tokens"]) == 2
    assert all(t.dtype == torch.long and t.numel() > 0 for t in batch["text_tokens"])
