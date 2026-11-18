# local-ai (Phase L1a, 512K): token embedding on the HOST instead of 2.54 GB BF16 on the card.
#
# ttnn.embedding only takes BF16 weights, so the table cannot shrink on the card. Instead the
# rows for the current tokens are gathered on the CPU and uploaded (decode: 10 KB/token, prefill
# chunk 2048: 21 MB). Drop-in for models.tt_transformers Embedding: self.embd(tok) keeps working
# at every call site, including traced paths:
#
#   * every uint32 token tensor that reaches the card through ttnn.from_torch(device=...),
#     ttnn.to_device or ttnn.copy_host_to_device_tensor is recorded (buffer address -> tokens),
#   * embd(tok) of a recorded buffer returns a clone of a PERSISTENT embedding buffer bound to that
#     address; the trace captures the clone, so replays read whatever is in the persistent buffer,
#   * a later copy_host_to_device_tensor into the token buffer (how traces get new inputs) also
#     rewrites the bound embedding buffer -> outside the trace, before the replay. No other code
#     changes needed.
#   * unrecorded tensors (eager only) are read back once; in trace capture that would fail loudly.
#
# Enabled with QWEN36_HOST_EMBED=1 (single device only).
import os

import torch
from loguru import logger

import ttnn

_ACTIVE = None  # the one HostEmbedding whose hooks are installed


def _addr(t):
    try:
        return int(t.buffer_address())
    except Exception:
        return None


def _as_tokens(t):
    """Token ids as ttnn.embedding sees them: rank 1 -> [S]; else [shape[0], shape[-1]]."""
    t = t.to(torch.int64)
    return t if t.dim() == 1 else t.reshape(t.shape[0], t.shape[-1])


def _nbytes(t):
    n = 1
    for d in t.shape:
        n *= int(d)
    return 2 * n


def _is_u32(t):
    try:
        return t.dtype == ttnn.uint32
    except Exception:
        return False


class HostEmbedding:
    def __init__(self, mesh_device, args, state_dict):
        base_name = args.get_state_dict_prefix("", None) + "tok_embeddings.weight"
        w = state_dict[base_name]
        self.weight = w.to(torch.bfloat16).contiguous()  # [vocab, H] stays in host RAM
        self.mesh_device = mesh_device
        self.vocab_size = args.vocab_size
        self.dim = self.weight.shape[-1]
        self._tok = {}  # buffer addr -> int64 token tensor as last written from host
        self._emb = {}  # buffer addr -> persistent device embedding tensor (TILE, bf16)
        self._bytes = 0
        _install_hooks(self)
        logger.info(
            f"[HOST_EMBED] embedding {tuple(self.weight.shape)} bf16 on host "
            f"({self.weight.numel() * 2 / 1e9:.2f} GB not on the card)"
        )

    # -- recording (called from the ttnn hooks) -------------------------------------------------
    def _record(self, dev, toks):
        a = _addr(dev)
        if a is None:
            return
        toks = _as_tokens(toks)
        self._tok[a] = toks
        e = self._emb.get(a)
        if e is None:
            return
        if tuple(e.shape)[:-1] == tuple(toks.shape):
            _ORIG["copy"](self._host_rows(toks), e)  # keep the trace input in step with the tokens
        else:  # a different tensor now lives at this address -> bound buffer is stale
            self._drop(a)

    def _drop(self, a):
        e = self._emb.pop(a, None)
        if e is not None:
            self._bytes -= _nbytes(e)
            ttnn.deallocate(e)

    # -- lookup ----------------------------------------------------------------------------------
    def _rows(self, toks):
        flat = toks.reshape(-1).clamp_(0, self.weight.shape[0] - 1)
        rows = self.weight.index_select(0, flat)
        if toks.dim() == 1:
            return rows.reshape(toks.shape[0], self.dim)
        return rows.reshape(toks.shape[0], toks.shape[-1], self.dim)

    def _host_rows(self, toks):
        return ttnn.from_torch(self._rows(toks), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def __call__(self, x, memory_config=None):
        return self.forward(x, memory_config=memory_config)

    def forward(self, x, memory_config=None):
        a = _addr(x)
        e = self._emb.get(a) if a is not None else None
        if e is not None:
            return ttnn.clone(e, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)
        toks = self._tok.get(a) if a is not None else None
        recorded = toks is not None
        if toks is None:  # eager fallback: tensor reached the card some other way
            toks = _as_tokens(ttnn.to_torch(x))
        dev = ttnn.from_torch(
            self._rows(toks),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        if not recorded:
            return dev
        # Recorded token buffer: may be (or become) a trace input -> keep a bound copy.
        self._emb[a] = dev
        self._bytes += _nbytes(dev)
        if len(self._emb) % 16 == 0:
            logger.info(f"[HOST_EMBED] {len(self._emb)} bound buffers, {self._bytes / 1e6:.1f} MB")
        return ttnn.clone(dev, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)


_ORIG = {}


def _install_hooks(he):
    global _ACTIVE
    _ACTIVE = he
    if _ORIG:
        return
    _ORIG["from_torch"] = ttnn.from_torch
    _ORIG["to_device"] = ttnn.to_device
    _ORIG["copy"] = ttnn.copy_host_to_device_tensor
    _ORIG["deallocate"] = ttnn.deallocate

    def from_torch(tensor, *args, **kwargs):
        out = _ORIG["from_torch"](tensor, *args, **kwargs)
        if _ACTIVE is not None and kwargs.get("device") is not None and _is_u32(out):
            try:
                _ACTIVE._record(out, tensor.detach().cpu())
            except Exception as ex:  # never break the caller
                logger.warning(f"[HOST_EMBED] record from_torch failed: {ex}")
        return out

    def to_device(tensor, *args, **kwargs):
        out = _ORIG["to_device"](tensor, *args, **kwargs)
        if _ACTIVE is not None and _is_u32(out):
            try:
                _ACTIVE._record(out, ttnn.to_torch(tensor))
            except Exception as ex:
                logger.warning(f"[HOST_EMBED] record to_device failed: {ex}")
        return out

    def copy_host_to_device_tensor(host, dev, *args, **kwargs):
        out = _ORIG["copy"](host, dev, *args, **kwargs)
        if _ACTIVE is not None and _is_u32(dev):
            _ACTIVE._record(dev, ttnn.to_torch(host))
        return out

    def deallocate(t, *args, **kwargs):
        if _ACTIVE is not None and _is_u32(t):
            a = _addr(t)
            if a is not None:
                _ACTIVE._tok.pop(a, None)
                _ACTIVE._drop(a)
        return _ORIG["deallocate"](t, *args, **kwargs)

    ttnn.from_torch = from_torch
    ttnn.to_device = to_device
    ttnn.copy_host_to_device_tensor = copy_host_to_device_tensor
    ttnn.deallocate = deallocate


def host_embed_enabled(args):
    return os.environ.get("QWEN36_HOST_EMBED", "0") == "1" and args.num_devices == 1
