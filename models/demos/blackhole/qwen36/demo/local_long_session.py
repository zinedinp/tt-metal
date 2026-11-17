# local-ai: long-session / long-context recall test for Qwen3.x-27B on ONE P150.
#
# Builds a realistic multi-turn chat that grows to ~N tokens:
#   turn 1   : the user states personal facts (codeword, number, project name)
#   middle   : many turns where the user pastes book sections and the assistant acknowledges;
#              one extra fact (the "needle") is hidden in a turn at ~50% depth
#   last turn: the user asks for all facts.
# The whole transcript is prefilled (chat template, thinking off) and the model must answer from
# context. PASS = every fact appears in the answer.
#
#   LONGSESSION_LENGTHS="8192,32768"   lengths to test (one pytest case each)
#   QWEN_SDPA_BF8=1                    bf8 paged KV (half the KV memory)
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import (
    _MESH_SHAPE,
    BLOCK_SIZE,
    DEVICE_PARAMS,
    _blocks_for,
    _run_traced_generation,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

DATA = Path(__file__).parent / "local_data"
FACTS = {
    "codeword": ("Mein geheimes Codewort lautet BLAUER-PELIKAN-47.", "BLAUER-PELIKAN-47"),
    "number": ("Meine Glückszahl ist 8317.", "8317"),
    "project": ("Mein Projekt heißt Holo-Maschine.", "Holo-Maschine"),
}
NEEDLE = ("Übrigens: der Server steht im Raum K-112.", "K-112")
LENGTHS = [int(x) for x in os.environ.get("LONGSESSION_LENGTHS", "8192,32768").split(",") if x.strip()]
MAX_NEW = int(os.environ.get("LONGSESSION_MAX_NEW", "160"))


def _corpus():
    text = (DATA / "pg84.txt").read_text() + "\n\n" + (DATA / "pg2600.txt").read_text()
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 200]


def _build_session(tokenizer, target_tokens):
    """Multi-turn chat of ~target_tokens; returns (token_ids [1,T], n_turns)."""
    sys_msg = "Du bist ein hilfreicher Assistent. Antworte knapp und auf Deutsch."
    first = " ".join(f for f, _ in FACTS.values()) + " Bitte merk dir das, ich frage später danach."
    question = (
        "Letzte Frage zu unserem ganzen Gespräch: Wie lautet mein geheimes Codewort, meine Glückszahl, "
        "der Name meines Projekts, und in welchem Raum steht der Server? Antworte in einer Zeile."
    )
    paras = _corpus()

    def render(msgs, gen):
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False
        )

    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": first},
        {"role": "assistant", "content": "Verstanden, ich habe mir alles gemerkt."},
    ]
    # Fill with book sections, ~1.5k tokens per user turn.
    budget = target_tokens - 300  # room for the question + template
    pi, needle_done, turns = 0, False, 0
    while True:
        chunk = []
        while sum(len(c) for c in chunk) < 6000 and pi < len(paras):
            chunk.append(paras[pi])
            pi += 1
        if not chunk:
            break
        content = "Hier ist der nächste Abschnitt zum Lesen:\n\n" + "\n\n".join(chunk)
        cur = len(tokenizer(render(msgs, False), add_special_tokens=False)["input_ids"])
        if not needle_done and cur >= budget // 2:
            content = NEEDLE[0] + "\n\n" + content
            needle_done = True
        trial = msgs + [{"role": "user", "content": content}, {"role": "assistant", "content": "Gelesen."}]
        if len(tokenizer(render(trial, False), add_special_tokens=False)["input_ids"]) > budget:
            break
        msgs, turns = trial, turns + 1
    if not needle_done:  # very short sessions: put the needle into the last filler turn
        msgs.insert(3, {"role": "user", "content": NEEDLE[0]})
        msgs.insert(4, {"role": "assistant", "content": "Notiert."})
    msgs.append({"role": "user", "content": question})
    ids = tokenizer(render(msgs, True), add_special_tokens=False, return_tensors="pt")["input_ids"]
    return ids, turns


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("target", LENGTHS, ids=[f"ctx{n // 1024}k" for n in LENGTHS])
@pytest.mark.timeout(7200)
def test_long_session(mesh_device, target):
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    num_blocks = _blocks_for(target, MAX_NEW)
    max_seq_len = num_blocks * BLOCK_SIZE

    t0 = time.time()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=max_seq_len, enable_mtp=False)
    logger.info(f"Model load: {time.time() - t0:.1f}s")
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)

    token_ids, turns = _build_session(tokenizer, target)
    T = token_ids.shape[1]
    kv = "bf8" if os.environ.get("QWEN_SDPA_BF8") == "1" else "bf16"
    logger.info(f"Session: {T} tokens, {turns} filler turns, KV {kv}, budget {num_blocks}x{BLOCK_SIZE}={max_seq_len}")

    generated, perf = _run_traced_generation(model, tokenizer, device, token_ids, MAX_NEW, num_blocks)
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    found = {k: (v[1] in answer) for k, v in {**FACTS, "needle": NEEDLE}.items()}

    mem = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    dram_used_gb = mem.total_bytes_allocated_per_bank * mem.num_banks / 1e9
    logger.info("=" * 70)
    logger.info(f"[LONGSESSION] ctx={T} kv={kv} ttft={perf['ttft']:.2f}s "
                f"decode={1.0 / perf['avg_decode_s']:.2f} tok/s dram_used={dram_used_gb:.1f}GB")
    logger.info(f"[LONGSESSION] recall={sum(found.values())}/{len(found)} {found}")
    logger.info(f"[LONGSESSION] answer: {answer.strip()[:400]}")
    logger.info("=" * 70)
    assert all(found.values()), f"recall failed at {T} tokens: {found}"


# ---------------------------------------------------------------------------------------------
# local-ai: DRAM inventory — where do the non-weight GB go? (weights on disk cache = 24.44 GB)
#   pytest ... -k inventory    INVENTORY_CTX=8192
# ---------------------------------------------------------------------------------------------
def _dram(device, tag, prev=None):
    mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    used = mv.total_bytes_allocated_per_bank * mv.num_banks
    sizes = {}
    for b in mv.block_table:
        if b.get("allocated") == "yes":
            s = int(b["size"]) * mv.num_banks
            sizes[s] = sizes.get(s, 0) + 1
    top = sorted(sizes.items(), key=lambda kv: -kv[0] * kv[1])[:12]
    delta = f" (+{(used - prev) / 1e9:.2f} GB)" if prev is not None else ""
    logger.info(f"[INVENTORY] {tag}: DRAM used {used / 1e9:.2f} GB{delta}, blocks {sum(sizes.values())}")
    for s, n in top:
        logger.info(f"[INVENTORY]     {n:5d} x {s / 1e6:9.2f} MB = {n * s / 1e9:6.2f} GB")
    return used


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.timeout(3600)
def test_inventory(mesh_device):
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    target = int(os.environ.get("INVENTORY_CTX", "8192"))
    num_blocks = _blocks_for(target, 32)
    u0 = _dram(device, "device open")
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE, enable_mtp=False)
    u1 = _dram(device, "after model load (weights expected 24.44 GB incl. embedding)", u0)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    token_ids, _ = _build_session(tokenizer, min(target, 4000))
    # KV for the full budget, as the generation path does
    kv_shape = [num_blocks, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    u2 = _dram(device, f"after KV alloc ({num_blocks} blocks, {os.environ.get('QWEN_SDPA_BF8', '0') == '1' and 'bf8' or 'bf16'})", u1)
    model.free_kv_caches()
    generated, perf = _run_traced_generation(model, tokenizer, device, token_ids, 16, num_blocks)
    u3 = _dram(device, "after traced prefill+decode (KV still allocated)", u1)
    logger.info(f"[INVENTORY] summary: weights+model {(u1 - u0) / 1e9:.2f} GB, KV {(u2 - u1) / 1e9:.2f} GB, "
                f"trace/prefill/decode buffers ≈ {(u3 - u2) / 1e9:.2f} GB, total {u3 / 1e9:.2f} GB")
