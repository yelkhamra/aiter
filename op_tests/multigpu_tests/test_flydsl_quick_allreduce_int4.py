# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness for FlyDSL INT4 quick all-reduce (``QuickAllReduceInt4``).

``python3`` this file first runs untimed codec/shape validity, then an
aiter-op-test ``@benchmark`` / markdown sweep. Every rank is a
``multiprocessing`` spawn worker that builds its own
``QuickAllReduceInt4`` engine, calls ``compile()``, and in the sweep
times ``fly.allreduce`` with ``run_perftest``. The oracle is an untimed
fp32 NCCL all-reduce of the same per-rank inputs. INT4 is lossy, so
validity uses SQNR, a calibrated mismatch ratio, and a per-tile SQNR
floor.

The divisor cases run at TP 8: with the E5M2 group scale an outlier fill
no longer clips at divisor 1, so divisor 1 must reach the SQNR floor and the
opt-in divisor 8 must not lose more than 1 dB against it; a small-magnitude
fill must keep full SQNR at the default divisor. An above-fp16 fill (partials
around 132k, reduced sums around 264k-334k, both signs, one large value per
group next to small ones) must stay finite at divisor 1, where it clips at
the fp16 range, and must reach the SQNR floor at divisor 8, which extends
the range to 65504 * 8. Every case also requires finite output.

The scale format is chosen per 512-value wave slice: E4M3 where every group
fits it, else E5M2. Data that fits E4M3 must keep E4M3 precision: the
N(0, 0.1) validity shapes and the in-range rows of two mixed fills (in-range
channel outliers; a few massive tokens among quiet rows) must reach
``IN_RANGE_SQNR_MIN_DB``, which an E5M2-only scale misses.

hidden=5120 is the width the kernel was tuned on, not a shape the kernel
requires. QuickAllReduceInt4 runs on gfx942/gfx950 at TP∈{2,4,8}; other
archs skip, and ``main()`` skips a world size when fewer GPUs are visible
than TP.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool, freeze_support, set_start_method

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

set_start_method("spawn", force=True)

from aiter.ops.flydsl.kernels.quick_allreduce_int4 import (
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WORLD,
)
from aiter.ops.flydsl.quick_allreduce_int4 import DEFAULT_DIVISOR, DEFAULT_GRID_CAP

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None
SUPPORTED_ARCHS = ("gfx942", "gfx950")
SQNR_MIN_DB = 18.0
# Data whose groups fit the E4M3 scale keeps E4M3 precision: about 19.2 dB
# on these fills, where an E5M2 scale gives 18.4 to 18.9.
IN_RANGE_SQNR_MIN_DB = 19.0
# A 32 KiB tile the kernel never wrote scores ~0 dB; codec noise stays above 8.
TILE_SQNR_MIN_DB = 8.0
# Calibrated to the INT4 group-16 codec vs fp32 all-reduce, not bit identity.
CLOSE_RTOL = 1e-1
CLOSE_ATOL = 1e-1
CLOSE_ERR_RATIO = 0.5
SUPER_TILE = 8
TP = WORLD
_FILLS = (
    "normal",
    "outlier",
    "small",
    "tiny",
    "pos_underflow",
    "neg_underflow",
    "overflow_512",
    "zeros",
    "large_every_rank",
    "above_fp16",
    "channel_outliers",
    "massive_tokens",
)
_CODEC_FILL_CASES = (
    ("pos_underflow", "pos-underflow-2^-8"),
    ("neg_underflow", "neg-underflow-2^-8"),
    ("overflow_512", "overflow-512"),
    ("zeros", "true-zero-scale"),
    ("large_every_rank", "every-rank-2048"),
    # Every group extremum between E5M2's 2**-12 and E4M3's 2**-7 minimum
    # scale: needs the E5M2 fallback for small slices.
    ("tiny", "tiny-std-1e-3"),
)
# (512, 5120) runs ST=1 and (32768, 5120) runs ST=8.
_DIVISOR_SHAPES = [(512, 5120), (32768, 5120)]
# Recommended opt-in divisor for inputs that clip at divisor 1.
_OUTLIER_DIVISOR = 8.0
# above_fp16: every 64th column is large, alternating sign by 64-column
# block. Ranks 0 and 1 send 132k and the others 11.7k, so the reduced sum is
# above the fp16 max at every TP size (~264k at TP 2, ~334k at TP 8) and
# below 65504 * _OUTLIER_DIVISOR.
_ABOVE_FP16_PERIOD = 64
_ABOVE_FP16_COL = 5
_ABOVE_FP16_LEAD = 132000.0
_ABOVE_FP16_REST = 11700.0
# Mixed-format fills, checked for in-range precision where they fit E4M3.
# channel_outliers: per-channel lognormal scale shared by every rank, sized
# so the reduced maximum stays near 400. massive_tokens: N(0, 1) with 2% of
# rows carrying 3k to 20k on 4 channels, so one call mixes E4M3 slices
# (quiet rows) and E5M2 slices (loud rows).
_MIXED_FILLS = ("channel_outliers", "massive_tokens")
_LOUD = 1000.0
# One spawn per world size so compile happens once. hidden=4096 is a width
# the ST pick was not fitted to; (8, 1024) is smaller than one 32 KiB tile.
_VALIDITY_SHAPES = {
    8: [(8, 1024), (512, 5120), (9216, 4096), (32768, 5120)],
    4: [(512, 5120), (9216, 4096)],
    2: [(512, 5120), (9216, 4096)],
}


def _make_inp(
    tokens: int,
    hidden: int,
    fill: str,
    *,
    rank: int,
    device: torch.device,
    world_size: int = WORLD,
) -> torch.Tensor:
    shape = (tokens, hidden)
    if fill == "channel_outliers":
        shared = torch.Generator().manual_seed(77)
        ch = torch.exp(torch.randn(hidden, generator=shared) * 1.5)
        ch = ch * (400.0 / (float(ch.max()) * 5.0 * world_size**0.5))
        gen = torch.Generator().manual_seed(1234 + rank)
        x = torch.randn(shape, generator=gen, dtype=torch.float32) * ch
        return x.to(device=device, dtype=torch.bfloat16)
    if fill == "massive_tokens":
        gen = torch.Generator().manual_seed(1234 + rank)
        x = torch.randn(shape, generator=gen, dtype=torch.float32)
        shared = torch.Generator().manual_seed(78)
        rows = torch.randperm(tokens, generator=shared)[: max(1, tokens // 50)]
        cols = torch.randperm(hidden, generator=shared)[:4]
        mag = torch.empty(len(rows), 4).uniform_(3e3, 2e4, generator=shared)
        sign = torch.where(torch.rand(len(rows), 4, generator=shared) < 0.5, -1.0, 1.0)
        x[rows[:, None], cols[None, :]] = sign * mag / world_size
        return x.to(device=device, dtype=torch.bfloat16)
    if fill in ("normal", "outlier", "small", "tiny"):
        gen = torch.Generator().manual_seed(1234 + rank)
        std = {"normal": 0.1, "outlier": 100.0, "small": 0.005, "tiny": 1e-3}[fill]
        return (torch.randn(shape, generator=gen, dtype=torch.float32) * std).to(
            device=device, dtype=torch.bfloat16
        )
    if fill == "above_fp16":
        gen = torch.Generator().manual_seed(1234 + rank)
        x = torch.randn(shape, generator=gen, dtype=torch.float32)
        cols = torch.arange(hidden)
        big = cols % _ABOVE_FP16_PERIOD == _ABOVE_FP16_COL
        sign = 1.0 - 2.0 * ((cols // _ABOVE_FP16_PERIOD) % 2).to(torch.float32)
        mag = _ABOVE_FP16_LEAD if rank < 2 else _ABOVE_FP16_REST
        x[:, big] = (sign * mag)[big]
        return x.to(device=device, dtype=torch.bfloat16)
    if fill == "pos_underflow":
        val = 2.0**-8
    elif fill == "neg_underflow":
        val = -(2.0**-8)
    elif fill == "overflow_512":
        # A rank-0 partial above 480, the largest E4M3 group scale; the E5M2
        # scale holds it. Only rank 0 carries the value.
        val = 512.0 if rank == 0 else 0.0
    elif fill == "large_every_rank":
        # Every partial and the reduced sum above 480: an E4M3 group scale
        # clipped both (MiniMax-M3 all-reduce outputs reach ~16k).
        val = 2048.0
    elif fill == "zeros":
        val = 0.0
    else:
        raise ValueError(f"unknown fill {fill!r}; expected one of {_FILLS}")
    return torch.full(shape, val, device=device, dtype=torch.bfloat16)


def _sqnr(ref_pow: torch.Tensor, mse: torch.Tensor) -> torch.Tensor:
    score = torch.where(
        (ref_pow <= 0) & (mse <= 0),
        torch.full_like(mse, float("inf")),
        10.0 * torch.log10(ref_pow / mse),
    )
    return torch.nan_to_num(score, nan=float("-inf"), neginf=float("-inf"))


def _sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        _sqnr((reference * reference).mean(), ((got - reference) ** 2).mean()).item()
    )


def _quiet_sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    """SQNR over rows whose reduced magnitude stays below ``_LOUD``."""
    quiet = reference.abs().amax(dim=-1) < _LOUD
    if not bool(quiet.any()):
        return float("inf")
    return _sqnr_db(got[quiet], reference[quiet])


def _min_tile_sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    """Worst 32 KiB-tile SQNR, so one unwritten tile cannot be averaged away."""
    tile_elems = TILE_BYTES // 2
    g = got.reshape(-1)
    r = reference.reshape(-1)
    n = int(g.numel())
    n_full = (n // tile_elems) * tile_elems
    vals = []
    if n_full:
        gt = g[:n_full].view(-1, tile_elems)
        rt = r[:n_full].view(-1, tile_elems)
        mse = ((gt - rt) ** 2).mean(dim=1)
        pow_ = (rt * rt).mean(dim=1)
        vals.append(float(_sqnr(pow_, mse).min().item()))
    if n > n_full:
        vals.append(_sqnr_db(g[n_full:], r[n_full:]))
    return min(vals) if vals else _sqnr_db(got, reference)


def _run_rank(
    rank: int,
    tp: int,
    init_method: str,
    tokens: list[int],
    hiddens: list[int],
    super_tile: int,
    grid_cap: int,
    fill: str,
    time_it: bool,
    divisor: float = DEFAULT_DIVISOR,
) -> list[dict]:
    import torch.distributed as dist

    from aiter.ops.flydsl import QuickAllReduceInt4

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=tp,
        rank=rank,
        device_id=device,
    )
    # QuickAllReduceInt4 exchanges IPC metadata over a non-NCCL group;
    # NCCL stays for the fp32 reference all-reduce.
    gloo = dist.new_group(backend="gloo")
    group = dist.group.WORLD

    fly = QuickAllReduceInt4(
        group=gloo,
        device=device,
        rank=rank,
        world_size=tp,
        super_tile=super_tile,
        grid_cap=grid_cap,
    )
    # compile() launches every ST binary on this shape and all ranks must pass
    # the same one, so keep the JIT buffer small at the widest hidden size.
    compile_tokens = min(512, max(tokens))
    compile_hidden = max(hiddens)
    compile_inp = torch.empty(
        (compile_tokens, compile_hidden), device=device, dtype=torch.bfloat16
    )
    compile_out = torch.empty_like(compile_inp)
    dist.barrier()
    fly.compile(compile_inp, compile_out)
    dist.barrier()
    del compile_inp, compile_out

    rows = []
    try:
        for ntok, hidden in zip(tokens, hiddens, strict=True):
            inp = _make_inp(ntok, hidden, fill, rank=rank, device=device, world_size=tp)
            ref = inp.to(torch.float32)
            dist.all_reduce(ref, group=group)
            dist.barrier()

            out = torch.empty_like(inp)
            fly.allreduce(inp, out, divisor=divisor)
            got = out.to(torch.float32)
            dist.barrier()

            close_err = checkAllclose(
                ref,
                got,
                rtol=CLOSE_RTOL,
                atol=CLOSE_ATOL,
                tol_err_ratio=CLOSE_ERR_RATIO,
                printLog=False,
                msg=f"quick_allreduce_int4 rank {rank}",
            )
            row = {
                "tokens": ntok,
                "hidden": hidden,
                "grid_cap": grid_cap,
                "sqnr_db": _sqnr_db(got, ref),
                "min_tile_sqnr_db": _min_tile_sqnr_db(got, ref),
                "err": float(close_err),
                "nonfinite": int((~torch.isfinite(got)).sum().item()),
                "quiet_sqnr_db": _quiet_sqnr_db(got, ref),
                "us": None,
            }
            if time_it:
                dist.barrier(group=group)
                torch.cuda.synchronize()

                def _allreduce(eng=fly, src=inp, dst=out, div=divisor):
                    eng.allreduce(src, dst, divisor=div)
                    return dst

                # use_cuda_event is mandatory here: run_perftest's default
                # timer wraps the iterations in torch.profiler, which collects
                # no device rows inside a spawn worker and then fails reducing
                # its empty trace. cuda.Event timing is unaffected.
                _, us = run_perftest(_allreduce, use_cuda_event=True)
                row["us"] = float(us)
            rows.append(row)
            del inp, out, ref, got
            torch.cuda.empty_cache()
    finally:
        fly.close()
        dist.destroy_process_group()
    return rows


def _spawn(
    world_size: int,
    pairs: list[tuple[int, int]],
    *,
    time_it: bool,
    super_tile: int = SUPER_TILE,
    grid_cap: int = DEFAULT_GRID_CAP,
    fill: str = "normal",
    divisor: float = DEFAULT_DIVISOR,
) -> list[list[dict]]:
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={world_size}")
    n_gpu = torch.cuda.device_count()
    if n_gpu < world_size:
        raise RuntimeError(f"QuickAllReduceInt4 needs {world_size} GPUs, have {n_gpu}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    token_list = [t for t, _ in pairs]
    hidden_list = [h for _, h in pairs]
    timeout = float(os.environ.get("FLYDSL_QR_TIMEOUT", "3600"))
    pool = Pool(processes=world_size)
    try:
        results = [
            pool.apply_async(
                _run_rank,
                kwds={
                    "rank": rank,
                    "tp": world_size,
                    "init_method": init_method,
                    "tokens": token_list,
                    "hiddens": hidden_list,
                    "super_tile": super_tile,
                    "grid_cap": grid_cap,
                    "fill": fill,
                    "time_it": time_it,
                    "divisor": divisor,
                },
            )
            for rank in range(world_size)
        ]
        ranks = [fut.get(timeout=timeout) for fut in results]
    except Exception:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    if len(ranks) != world_size:
        raise RuntimeError(
            f"QuickAllReduceInt4 gathered {len(ranks)} ranks, expected {world_size}"
        )
    return ranks


def _assert_validity(
    ranks: list[list[dict]],
    *,
    pairs: list[tuple[int, int]],
    world_size: int,
    label: str,
    check_close: bool = True,
    check_sqnr: bool = True,
    sqnr_min: float = SQNR_MIN_DB,
) -> dict:
    if len(ranks) != world_size:
        raise AssertionError(
            f"{label}: gathered {len(ranks)} ranks, expected {world_size}"
        )
    fails = []
    for rank, rows in enumerate(ranks):
        if len(rows) != len(pairs):
            fails.append(f"rank {rank}: {len(rows)} rows, expected {len(pairs)}")
            continue
        for (tokens, hidden), row in zip(pairs, rows, strict=True):
            where = f"rank {rank} tokens={tokens} hidden={hidden}"
            if row["nonfinite"]:
                fails.append(f"{where}: {row['nonfinite']} non-finite outputs")
            if not check_sqnr:
                continue
            if row["sqnr_db"] < sqnr_min:
                fails.append(f"{where}: SQNR {row['sqnr_db']:.2f} dB < {sqnr_min}")
            if row["min_tile_sqnr_db"] < TILE_SQNR_MIN_DB:
                fails.append(
                    f"{where}: min-tile SQNR {row['min_tile_sqnr_db']:.2f} dB "
                    f"< {TILE_SQNR_MIN_DB}"
                )
            if check_close and row["err"] >= CLOSE_ERR_RATIO:
                fails.append(
                    f"{where}: checkAllclose err {row['err']:.3f} >= {CLOSE_ERR_RATIO}"
                )
    if fails:
        raise AssertionError(f"{label} tp={world_size}: " + "; ".join(fails))
    return ranks[0][0]


def _assert_above_fp16(
    world_size: int, pairs: list[tuple[int, int]], *, grid_cap: int
) -> None:
    """Reduced sums above the fp16 max: finite at divisor 1, accurate at 8."""
    for divisor in (1.0, _OUTLIER_DIVISOR):
        ranks = _spawn(
            world_size,
            pairs,
            time_it=False,
            fill="above_fp16",
            grid_cap=grid_cap,
            divisor=divisor,
        )
        aiter.logger.info(
            "above-fp16 tp=%s divisor=%s: SQNR %s dB",
            world_size,
            divisor,
            [round(min(r[k]["sqnr_db"] for r in ranks), 2) for k in range(len(pairs))],
        )
        # Divisor 1 saturates at the fp16 max by design; only finiteness holds.
        # CLOSE_ATOL is calibrated to the 0.1-std fill, not to these values.
        _assert_validity(
            ranks,
            pairs=pairs,
            world_size=world_size,
            label=f"above-fp16 divisor={divisor}",
            check_close=False,
            check_sqnr=divisor != 1.0,
        )


def _assert_in_range_precision(
    world_size: int, pairs: list[tuple[int, int]], *, grid_cap: int
) -> None:
    """Rows that fit the E4M3 group scale keep E4M3 precision.

    The scale format is chosen per 512-value wave slice, so quiet rows next
    to loud ones (and in-range channel outliers) must not fall back to the
    coarser E5M2 mantissa.
    """
    for fill in _MIXED_FILLS:
        ranks = _spawn(world_size, pairs, time_it=False, fill=fill, grid_cap=grid_cap)
        _assert_validity(
            ranks,
            pairs=pairs,
            world_size=world_size,
            label=fill,
            check_close=False,
        )
        fails = []
        for rank, rows in enumerate(ranks):
            for (tokens, hidden), row in zip(pairs, rows, strict=True):
                if row["quiet_sqnr_db"] < IN_RANGE_SQNR_MIN_DB:
                    fails.append(
                        f"rank {rank} tokens={tokens} hidden={hidden}: in-range "
                        f"SQNR {row['quiet_sqnr_db']:.2f} dB < {IN_RANGE_SQNR_MIN_DB}"
                    )
        aiter.logger.info(
            "%s tp=%s: in-range SQNR %s dB",
            fill,
            world_size,
            [
                round(min(r[k]["quiet_sqnr_db"] for r in ranks), 2)
                for k in range(len(pairs))
            ],
        )
        if fails:
            raise AssertionError(f"{fill} tp={world_size}: " + "; ".join(fails))


@benchmark()
def test_quick_allreduce_int4(tokens, hidden, dtype, tp, grid_cap=DEFAULT_GRID_CAP):
    ranks = _spawn(tp, [(tokens, hidden)], time_it=True, grid_cap=grid_cap)
    row = _assert_validity(
        ranks,
        pairs=[(tokens, hidden)],
        world_size=tp,
        label="bench",
    )
    nbytes = tokens * hidden * 2
    # (tp - 1) adds per element; codec ALU work is not counted.
    flops = tokens * hidden * (tp - 1)
    rank_us = [r[0]["us"] for r in ranks]
    us = max(rank_us)
    return {
        "gfx": ARCH,
        "tp": tp,
        "grid_cap": row["grid_cap"],
        "flydsl us": us,
        "flydsl TFLOPS": (flops / us / 1e6) if us else 0.0,
        "flydsl TB/s": (nbytes / us / 1e6) if us else 0.0,
        "flydsl err": max(r[0]["err"] for r in ranks),
        "flydsl sqnr_db": min(r[0]["sqnr_db"] for r in ranks),
    }


test_quick_allreduce_int4.__test__ = False


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("QuickAllReduceInt4 unsupported on %s; skipping", ARCH)
        return
    n_gpu = torch.cuda.device_count()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        help="Payload dtype (bf16 only).\n    e.g.: -d bf16",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1],
        help="Not a QuickAllReduceInt4 dimension; only 1 runs, other values are skipped.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="*",
        default=[TP],
        help="World sizes to sweep (2, 4, or 8). Default 8.\n    e.g.: --tp 8",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (512, 5120),
            (9216, 5120),
            (32768, 5120),
        ],
        help="(tokens, hidden) pairs; hidden is free, 5120 is the tuned width.\n"
        "    e.g.: -s 512,5120 9216,5120",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Optional JSON output path for the sweep rows.",
    )
    parser.add_argument(
        "--grid-cap",
        type=int,
        default=DEFAULT_GRID_CAP,
        help="Persistent-launch block cap; the engine clamps it to the\n"
        "    measured resident workgroups per CU.",
    )
    args = parser.parse_args()

    if n_gpu < 2:
        aiter.logger.warning(
            "QuickAllReduceInt4 needs 2 GPUs for codec fills, have %s; skipping",
            n_gpu,
        )
    else:
        codec_pairs = [(16, 1024)]
        for fill, label in _CODEC_FILL_CASES:
            ranks = _spawn(
                2, codec_pairs, time_it=False, fill=fill, grid_cap=args.grid_cap
            )
            _assert_validity(ranks, pairs=codec_pairs, world_size=2, label=label)
        _assert_above_fp16(2, codec_pairs, grid_cap=args.grid_cap)
        _assert_in_range_precision(2, [(512, 5120)], grid_cap=args.grid_cap)

    for tp, pairs in _VALIDITY_SHAPES.items():
        if n_gpu < tp:
            aiter.logger.warning(
                "QuickAllReduceInt4 needs %s GPUs, have %s; skipping validity tp=%s",
                tp,
                n_gpu,
                tp,
            )
            continue
        ranks = _spawn(tp, pairs, time_it=False, grid_cap=args.grid_cap)
        # N(0, 0.1) fits the E4M3 group scale everywhere.
        _assert_validity(
            ranks,
            pairs=pairs,
            world_size=tp,
            label="validity",
            sqnr_min=IN_RANGE_SQNR_MIN_DB,
        )

    if n_gpu < TP:
        aiter.logger.warning(
            "QuickAllReduceInt4 needs %s GPUs for divisor cases, have %s; skipping",
            TP,
            n_gpu,
        )
    else:
        ranks = _spawn(
            TP, _DIVISOR_SHAPES, time_it=False, fill="small", grid_cap=args.grid_cap
        )
        _assert_validity(
            ranks, pairs=_DIVISOR_SHAPES, world_size=TP, label="small default divisor"
        )

        outlier = {}
        for divisor in (1.0, _OUTLIER_DIVISOR):
            outlier[divisor] = _spawn(
                TP,
                _DIVISOR_SHAPES,
                time_it=False,
                fill="outlier",
                grid_cap=args.grid_cap,
                divisor=divisor,
            )
        # CLOSE_ATOL is calibrated to the 0.1-std fill, not to std 100.
        _assert_validity(
            outlier[_OUTLIER_DIVISOR],
            pairs=_DIVISOR_SHAPES,
            world_size=TP,
            label=f"outlier divisor={_OUTLIER_DIVISOR}",
            check_close=False,
        )
        for k, (tokens, hidden) in enumerate(_DIVISOR_SHAPES):
            base = min(r[k]["sqnr_db"] for r in outlier[1.0])
            fixed = min(r[k]["sqnr_db"] for r in outlier[_OUTLIER_DIVISOR])
            aiter.logger.info(
                "outlier tokens=%s hidden=%s: SQNR %.2f dB at divisor 1, "
                "%.2f dB at divisor %s",
                tokens,
                hidden,
                base,
                fixed,
                _OUTLIER_DIVISOR,
            )
            if base < SQNR_MIN_DB:
                raise AssertionError(
                    f"outlier tokens={tokens} hidden={hidden}: SQNR {base:.2f} dB "
                    f"at divisor 1 < {SQNR_MIN_DB}; the group scale clips"
                )
            if fixed < base - 1.0:
                raise AssertionError(
                    f"outlier tokens={tokens} hidden={hidden}: divisor "
                    f"{_OUTLIER_DIVISOR} gives {fixed:.2f} dB vs {base:.2f} dB "
                    "at divisor 1; the opt-in divisor should not cost precision"
                )
        _assert_above_fp16(TP, _DIVISOR_SHAPES, grid_cap=args.grid_cap)
        _assert_in_range_precision(TP, _DIVISOR_SHAPES, grid_cap=args.grid_cap)

    for dtype in args.dtype:
        if dtype != dtypes.bf16:
            aiter.logger.warning(
                "QuickAllReduceInt4 payload is bf16; skipping %s", dtype
            )
            continue
        df = []
        for tp, batch, mnk in itertools.product(args.tp, args.batch, args.mnk):
            if batch != 1:
                continue
            if tp not in SUPPORTED_WORLDS:
                aiter.logger.warning(
                    "QuickAllReduceInt4 unsupported world_size=%s; skipping", tp
                )
                continue
            if n_gpu < tp:
                aiter.logger.warning(
                    "QuickAllReduceInt4 needs %s GPUs, have %s; skipping tp=%s",
                    tp,
                    n_gpu,
                    tp,
                )
                continue
            if not isinstance(mnk, tuple) or len(mnk) != 2:
                raise ValueError(f"-s expects tokens,hidden; got {mnk!r}")
            tokens, hidden = int(mnk[0]), int(mnk[1])
            df.append(
                test_quick_allreduce_int4(
                    tokens, hidden, dtype, tp, grid_cap=args.grid_cap
                )
            )
        if df:
            table = pd.DataFrame(df)
            aiter.logger.info(
                "flydsl quick allreduce INT4 summary (markdown):\n%s",
                table.to_markdown(index=False),
            )
            if args.out:
                out_dir = os.path.dirname(os.path.abspath(args.out))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(args.out, "w") as fh:
                    json.dump(
                        {
                            "meta": {
                                "gfx": ARCH,
                                "grid_cap": args.grid_cap,
                                "timer": "run_perftest cuda_event",
                            },
                            "rows": df,
                        },
                        fh,
                        indent=2,
                        default=str,
                    )
                aiter.logger.info("wrote %s", args.out)


if __name__ == "__main__":
    freeze_support()
    main()
