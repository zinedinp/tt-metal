# local-ai: Q-A – Teacher-Forcing-Messung gegen die GGUF-Q6_K-Referenz (llama.cpp, CPU).
#
# Liest die Token aus der Referenzdatei von `llama-perplexity --kl-divergence-base` (gleiche Token,
# gleiche Chunks), rechnet jeden Chunk mit dem kurzen Prefill (T <= 1024, wie prefill_paged) und
# den LM-Head für ALLE gewerteten Positionen (zweite Chunk-Hälfte) – nicht nur für das letzte Token.
# Ausgabe im selben Format wie die Referenz; verglichen wird in Rust (codeki-qa compare).
#
#   QA_BASE="q6k-prose.kld,q6k-code.kld"   Referenzdateien in /root/.cache/qa
#   QA_TAG=base                             Name des Kandidaten (tt-<tag>-<text>.kld)
#   QA_LOGITS=fp32                          fp32 (misst die Gewichte) | bf16 (wie der Decode-Pfad)
#   QA_BOS=<id>                             nur falls die Referenz BOS an den Chunkanfang setzt
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import Mode

QA_DIR = Path(os.environ.get("QA_DIR", "/root/.cache/qa"))
BASES = [b for b in os.environ.get("QA_BASE", "q6k-prose.kld,q6k-code.kld").split(",") if b.strip()]
TAG = os.environ.get("QA_TAG", "base")
LOGITS = os.environ.get("QA_LOGITS", "fp32")
BOS = os.environ.get("QA_BOS")
ROWS = 128  # LM-Head-Stück (128 x 248320 fp32 = 127 MB)


def _read_base(path):
    raw = path.read_bytes()
    assert raw[:8] == b"_logits_", f"{path}: kein _logits_-Kopf"
    n_ctx, n_vocab, n_chunk = np.frombuffer(raw, dtype="<i4", count=3, offset=8)
    n_ctx, n_vocab, n_chunk = int(n_ctx), int(n_vocab), int(n_chunk)
    head = raw[: 20 + 4 * n_ctx * n_chunk]
    tokens = np.frombuffer(head, dtype="<i4", offset=20).reshape(n_chunk, n_ctx)
    return n_ctx, n_vocab, n_chunk, head, tokens


def _encode(logits, n_vocab):
    """Wie llama.cpp log_softmax(…, uint16_t*): Logprobs in [max-16, max] auf u16, plus scale/min (f32)."""
    l = logits[:, :n_vocab].double()
    mx = l.max(dim=-1, keepdim=True).values
    mn = torch.maximum(l.min(dim=-1, keepdim=True).values, mx - 16)
    lse = torch.log(torch.exp(l - mx).sum(dim=-1, keepdim=True))
    minlp = (mn - mx - lse).float()
    scale = ((mx - mn) / 65535.0).float()
    q = torch.where(l > mn, torch.round((l - mn) / scale.double().clamp_min(1e-30)), torch.zeros_like(l))
    q = q.clamp(0, 65535).to(torch.int32).numpy().astype("<u2")
    nv = 2 * ((n_vocab + 1) // 2) + 4
    rec = np.zeros((l.shape[0], nv), dtype="<u2")
    rec[:, 0:2] = scale.numpy().astype("<f4").view("<u2").reshape(-1, 2)
    rec[:, 2:4] = minlp.numpy().astype("<f4").view("<u2").reshape(-1, 2)
    rec[:, 4 : 4 + n_vocab] = q
    return rec


def _chunk_logprobs(model, ids, first, n_vocab, out_dtype):
    """Kurzer Prefill (identisch zu prefill_paged, T<=1024), dann LM-Head auf Positionen first..T-2."""
    B, T = ids.shape
    model._build_request_rope(ids, None)
    model.reset_state()
    tok = ttnn.from_torch(ids, dtype=ttnn.uint32, device=model.device)
    x = model.embd(tok)
    ttnn.deallocate(tok)
    cos, sin = model.rope.get_prefill_rot_mats(0, T)
    for layer in model.layers:
        x = layer.forward(x, cos=cos, sin=sin, mode="prefill")
    x = model.norm(x, mode=Mode.PREFILL)
    recs = []
    for a in range(first, T, ROWS):  # first und T sind Vielfache von 32
        b = min(a + ROWS, T)
        xs = x[:, a:b, :]
        lg = model._lm_head(xs, out_dtype=out_dtype)
        t = ttnn.to_torch(lg).float().reshape(b - a, -1)
        ttnn.deallocate(lg)
        ttnn.deallocate(xs)
        recs.append(_encode(t, n_vocab))
    ttnn.deallocate(x)
    rec = np.concatenate(recs, axis=0)
    return rec[: T - 1 - first]  # Position T-1 hat kein Folgetoken


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.timeout(7200)
def test_qa(mesh_device):
    device = mesh_device
    device.enable_program_cache()
    out_dtype = ttnn.float32 if LOGITS == "fp32" else None
    bases = [(b, *_read_base(QA_DIR / b)) for b in BASES]
    n_ctx = max(b[1] for b in bases)
    assert n_ctx <= 1024 and n_ctx % 64 == 0, f"n_ctx={n_ctx}: kurzer Prefill braucht <=1024, Vielfaches von 64"

    t0 = time.time()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=2048, enable_mtp=False)
    logger.info(f"Model load: {time.time() - t0:.1f}s")

    for name, n_ctx, n_vocab, n_chunk, head, tokens in bases:
        first = n_ctx // 2
        dst = QA_DIR / f"tt-{TAG}-{name.removeprefix('q6k-')}"
        tmp = dst.with_suffix(".tmp")
        t1 = time.time()
        with open(tmp, "wb") as f:
            f.write(head)
            for c in range(n_chunk):
                ids = torch.from_numpy(tokens[c].astype(np.int64)).unsqueeze(0).clone()
                if BOS is not None:
                    ids[0, 0] = int(BOS)
                rec = _chunk_logprobs(model, ids, first, n_vocab, out_dtype)
                f.write(rec.tobytes())
                logger.info(f"[QA] {name} Chunk {c + 1}/{n_chunk} ({time.time() - t1:.1f}s)")
        tmp.rename(dst)
        logger.info(f"[QA] {dst.name}: {n_chunk}x{n_ctx - 1 - first} Positionen, logits={LOGITS}, "
                    f"{time.time() - t1:.1f}s")
