# local-ai (Phase H3): decode-path quality without a GGUF reference -- the same tokens once through
# prefill (GDN chunk kernel in FP32, the trusted path) and once token by token through decode
# (teacher forcing). KL(prefill || decode) and top-1 agreement per position show what the decode
# path (fused GDN kernel or the composite, QWEN36_GDN_FUSED=1/0) loses.
#   CONS_TOKENS=256   positions compared (from chunk 0 of cache/qa/q8-prose.kld)
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, BLOCK_SIZE
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import Mode

N = int(os.environ.get("CONS_TOKENS", "256"))
QA_DIR = Path(os.environ.get("QA_DIR", "/root/.cache/qa"))


def _tokens():
    raw = (QA_DIR / "q8-prose.kld").read_bytes()
    n_ctx, _, _ = np.frombuffer(raw, dtype="<i4", count=3, offset=8)
    return torch.from_numpy(np.frombuffer(raw, dtype="<i4", count=int(n_ctx), offset=20).astype(np.int64))


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "num_command_queues": 2, "trace_region_size": 50_000_000}], indirect=True
)
@pytest.mark.timeout(7200)
def test_decode_consistency(mesh_device):
    dev = mesh_device
    dev.enable_program_cache()
    ids = _tokens()[: N + 1].unsqueeze(0)
    T = 512  # like local_qa chunks; other lengths hit the GDN-seq L1 clash (program 244)
    assert N < T
    model = Qwen36Model.from_pretrained(dev, max_batch_size=1, max_seq_len=2048, enable_mtp=False)
    num_blocks = 2048 // BLOCK_SIZE
    model.allocate_kv_caches([num_blocks, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim], ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    # 1) prefill reference: logits of positions 0..N-1 (predicting tokens 1..N)
    pad = torch.zeros(1, T, dtype=torch.int64)
    pad[0, : N + 1] = ids[0]
    model._build_request_rope(pad, None)
    model.reset_state()
    tok = ttnn.from_torch(pad, dtype=ttnn.uint32, device=dev)
    x = model.embd(tok)
    cos, sin = model.rope.get_prefill_rot_mats(0, T)
    for layer in model.layers:
        x = layer.forward(x, cos=cos, sin=sin, mode="prefill")
    x = model.norm(x, mode=Mode.PREFILL)
    ref = []
    for a in range(0, N, 128):
        b = min(a + 128, N)
        lg = model._lm_head(x[:, a:b, :], out_dtype=ttnn.float32)
        ref.append(ttnn.to_torch(lg).float().reshape(b - a, -1))
        ttnn.deallocate(lg)
    ttnn.deallocate(x)
    ref = torch.cat(ref)

    # 2) decode, teacher-forced, from an empty state
    model.reset_state()
    t0 = time.time()
    got = []
    for p in range(N):
        d = model.prepare_inputs_decode(ids[:, p : p + 1], p, page_table=page_table)
        c, s = unpack_rope(d[2])
        lg = model._forward_decode(d[0], c, s, d[1], d[3])
        got.append(ttnn.to_torch(lg).float().reshape(-1, lg.shape[-1])[0])
        ttnn.deallocate(lg)
    got = torch.stack(got)
    V = min(ref.shape[-1], got.shape[-1], 248320)
    lp_ref = torch.log_softmax(ref[:, :V].double(), -1)
    lp_got = torch.log_softmax(got[:, :V].double(), -1)
    kl = (lp_ref.exp() * (lp_ref - lp_got)).sum(-1)
    top1 = (lp_ref.argmax(-1) == lp_got.argmax(-1)).float()
    tag = "fused" if os.environ.get("QWEN36_GDN_FUSED", "1") == "1" else "composite"
    q = lambda t, a, b: t[a:b].mean().item()
    logger.info(
        f"[CONS] {tag}: {N} decode steps in {time.time() - t0:.0f}s | KL(prefill||decode) mean {kl.mean():.5f}, "
        f"p99 {kl.quantile(0.99):.4f}, max {kl.max():.4f} | top1 agree {100 * top1.mean():.2f} % | "
        f"KL by quarter {q(kl, 0, N // 4):.5f} {q(kl, N // 4, N // 2):.5f} {q(kl, N // 2, 3 * N // 4):.5f} {q(kl, 3 * N // 4, N):.5f}"
    )
