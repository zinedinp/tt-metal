# local-ai: Stack-Diagnose – jede Teilschicht der Karte gegen eine FP32-CPU-Referenz (HF transformers, BF16-Gewichte
# aus den safetensors), jeweils MIT DER EINGABE DER KARTE (teacher-forced pro Schicht). So zeigt jede Zeile den Fehler,
# den genau dieser Baustein neu einbringt – ohne die Summe aller Fehler davor.
#
# Pro Schicht:  mixer = Attention (jede 4.) bzw. Gated DeltaNet,  mlp = SwiGLU-MLP,  dazu am Ende norm+LM-Head.
#   rel  = |Karte - Ref| / |Ref|            (Fehler relativ zum Beitrag des Bausteins)
#   cos  = Kosinus Karte/Ref
#   inj  = |Karte - Ref| / |Residuum|       (wie stark der Fehler den Residualstrom verschiebt)
#
#   QA_DIR/QA_BASE   Token-Quelle (Default q8-prose.kld),  LD_CHUNK=0  welcher Chunk,  LD_T=1024 Token (wie Q-A; <=512 nimmt einen anderen L1-Pfad)
#   LD_LAYERS=0-63   Schichtbereich,  LD_OUT=/root/.cache/qa/layerdiff-<tag>.json
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import Mode

QA_DIR = Path(os.environ.get("QA_DIR", "/root/.cache/qa"))
BASE = os.environ.get("QA_BASE", "q8-prose.kld")
CHUNK = int(os.environ.get("LD_CHUNK", "0"))
T = int(os.environ.get("LD_T", "1024"))
TAG = os.environ.get("QA_TAG", "base")
_lo, _, _hi = os.environ.get("LD_LAYERS", "0-63").partition("-")
LAYERS = range(int(_lo), int(_hi or _lo) + 1)
OUT = Path(os.environ.get("LD_OUT", str(QA_DIR / f"layerdiff-{TAG}.json")))
# LD_DUMP=<Ordner>: je Schicht L<i>.safetensors (Karte + torch-Wahrheit) für codeki-ref schreiben.
DUMP = Path(os.environ["LD_DUMP"]) if os.environ.get("LD_DUMP") else None
PREFIX = "model.language_model."


def _tokens():
    raw = (QA_DIR / BASE).read_bytes()
    assert raw[:8] == b"_logits_"
    n_ctx, _, n_chunk = np.frombuffer(raw, dtype="<i4", count=3, offset=8)
    tok = np.frombuffer(raw, dtype="<i4", count=int(n_ctx) * int(n_chunk), offset=20).reshape(int(n_chunk), int(n_ctx))
    return torch.from_numpy(tok[CHUNK, :T].astype(np.int64)).unsqueeze(0).clone()


class _Weights:
    """Lädt nur die Tensoren einer Schicht (mmap, BF16 → FP32)."""

    def __init__(self, root):
        self.root = Path(root)
        self.map = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, prefix):
        out, files = {}, {}
        for k, f in self.map.items():
            if k.startswith(prefix):
                files.setdefault(f, []).append(k)
        for f, keys in files.items():
            with safe_open(str(self.root / f), framework="pt") as h:
                for k in keys:
                    out[k[len(prefix):]] = h.get_tensor(k).float()
        return out


def _host(t, H):
    return ttnn.to_torch(t).float().reshape(-1, H)[:T].clone()


def _m(card, ref, resid):
    d = (card - ref).double()
    nr = ref.double().norm().item()
    return {
        "rel": d.norm().item() / max(nr, 1e-30),
        "cos": torch.nn.functional.cosine_similarity(card.double().flatten(), ref.double().flatten(), dim=0).item(),
        "inj": d.norm().item() / max(resid.double().norm().item(), 1e-30),
        "maxabs": d.abs().max().item(),
    }


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.timeout(7200)
def test_layerdiff(mesh_device):
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    torch.set_grad_enabled(False)
    root = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(root)
    cfg = getattr(cfg, "text_config", cfg)
    cfg._attn_implementation = "eager"
    H = cfg.hidden_size
    W = _Weights(root)

    ids = _tokens()
    pos = torch.arange(T)[None]
    rope = hf.Qwen3_5TextRotaryEmbedding(cfg)
    pe = rope(torch.zeros(1, T, H), pos)
    mask = torch.full((T, T), float("-inf")).triu(1)[None, None]

    device = mesh_device
    device.enable_program_cache()
    t0 = time.time()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=2048, enable_mtp=False)
    logger.info(f"Model load: {time.time() - t0:.1f}s, Token {ids.shape[1]} aus {BASE} Chunk {CHUNK}")

    model._build_request_rope(ids, None)
    model.reset_state()
    tok = ttnn.from_torch(ids, dtype=ttnn.uint32, device=device)
    x = model.embd(tok)
    ttnn.deallocate(tok)
    emb_ref = W.get(PREFIX + "embed_tokens.")["weight"][ids[0]]
    rows = [{"layer": -1, "kind": "embed", **_m(_host(x, H), emb_ref, emb_ref)}]
    logger.info(f"[LD] embed rel={rows[0]['rel']:.2e}")
    cos, sin = model.rope.get_prefill_rot_mats(0, T)

    for i, layer in enumerate(model.layers):
        rec = {}
        if i in LAYERS:
            # Ausgaben der Teilblöcke mitschneiden (nur lesen; Tensoren bleiben auf der Karte unverändert).
            fa, ff = layer.attention.forward, layer.feed_forward.forward

            def wa(*a, _f=fa, **k):
                o = _f(*a, **k)
                rec["mixer"] = _host(o[0] if isinstance(o, tuple) else o, H)
                return o

            def wf(*a, _f=ff, **k):
                o = _f(*a, **k)
                rec["mlp"] = _host(o, H)
                return o

            layer.attention.forward, layer.feed_forward.forward = wa, wf
            x_in = _host(x, H)
        y = layer.forward(x, cos=cos, sin=sin, mode="prefill")
        if i not in LAYERS:
            x = y
            continue
        layer.attention.forward, layer.feed_forward.forward = fa, ff
        x_out = _host(y, H)
        x = y

        ref = hf.Qwen3_5DecoderLayer(cfg, i).float().eval()
        missing, unexpected = ref.load_state_dict(W.get(f"{PREFIX}layers.{i}."), strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        kind = ref.layer_type
        h_in = ref.input_layernorm(x_in[None])
        if kind == "linear_attention":
            mix_ref = ref.linear_attn(hidden_states=h_in, attention_mask=None)[0]
        else:
            mix_ref = ref.self_attn(hidden_states=h_in, position_embeddings=pe, attention_mask=mask)[0][0]
        mix_card = rec.get("mixer")
        # MLP mit der Karten-Mitte (x_in + Karten-Mixer), damit nur der MLP-Fehler zählt.
        h_mid = x_in + (mix_card if mix_card is not None else mix_ref)
        mlp_ref = ref.mlp(ref.post_attention_layernorm(h_mid[None]))[0]
        mlp_card = rec.get("mlp")
        out_ref = h_mid + mlp_ref
        r = {"layer": i, "kind": "attn" if kind == "full_attention" else "gdn"}
        if mix_card is not None:
            r["mixer"] = _m(mix_card, mix_ref, x_in)
        if mlp_card is not None:
            r["mlp"] = _m(mlp_card, mlp_ref, h_mid)
        r["layer_out"] = _m(x_out - x_in, out_ref - x_in, x_in)
        r["resid_norm"] = x_in.norm(dim=-1).mean().item()
        rows.append(r)
        if DUMP is not None:
            # Für codeki-ref (Rust-Referenz): Karteneingabe/-ausgaben + torch-Wahrheit, FP32, [T, H].
            DUMP.mkdir(parents=True, exist_ok=True)
            save_file(
                {
                    "x_in": x_in.contiguous(),
                    "h_mid": h_mid.contiguous(),
                    "x_out": x_out.contiguous(),
                    "mix_card": mix_card.contiguous(),
                    "mlp_card": mlp_card.contiguous(),
                    "mix_torch": mix_ref.contiguous(),
                    "mlp_torch": mlp_ref.contiguous(),
                    "ids": ids[0].to(torch.int32).contiguous(),
                },
                str(DUMP / f"L{i:02d}.safetensors"),
                metadata={"kind": kind, "layer": str(i), "T": str(T)},
            )
        mx, ml = r.get("mixer", {}), r.get("mlp", {})
        logger.info(
            f"[LD] L{i:02d} {r['kind']:4s} mixer rel={mx.get('rel', float('nan')):.3e} cos={mx.get('cos', float('nan')):.5f} "
            f"inj={mx.get('inj', float('nan')):.2e} | mlp rel={ml.get('rel', float('nan')):.3e} "
            f"cos={ml.get('cos', float('nan')):.5f} inj={ml.get('inj', float('nan')):.2e} | |x|={r['resid_norm']:.1f}"
        )
        del ref

    if LAYERS.stop == len(model.layers):
        xf = _host(x, H)
        xn = model.norm(x, mode=Mode.PREFILL)
        n = 64
        lg = model._lm_head(xn[:, T - n : T, :], out_dtype=ttnn.float32)
        card_lg = ttnn.to_torch(lg).float().reshape(n, -1)[:, : cfg.vocab_size]
        norm_ref = hf.Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        norm_ref.load_state_dict(W.get(PREFIX + "norm."))
        head = W.get("lm_head.")["weight"]
        ref_lg = norm_ref(xf[T - n :][None])[0] @ head.T
        r = {"layer": 64, "kind": "norm+lm_head", "logits": _m(card_lg, ref_lg, ref_lg)}
        r["top1_same"] = (card_lg.argmax(-1) == ref_lg.argmax(-1)).float().mean().item()
        lp_c, lp_r = card_lg.log_softmax(-1), ref_lg.log_softmax(-1)
        r["kld"] = (lp_r.exp() * (lp_r - lp_c)).sum(-1).mean().item()
        rows.append(r)
        logger.info(f"[LD] norm+lm_head rel={r['logits']['rel']:.3e} top1={r['top1_same']:.3f} KLD={r['kld']:.4f}")

    OUT.write_text(json.dumps({"base": BASE, "chunk": CHUNK, "T": T, "tag": TAG, "rows": rows}, indent=1))
    logger.info(f"[LD] → {OUT}")
