# okc: problem=muse_bf16_proj
"""TileLang projections for H200.

One-token steps use a vectorized GEMV. Other steps use pipelined BF16
Tensor Core GEMMs, splitting K when the output grid is too small. Partial
sums stay FP32 and are reduced in a fixed order. The router uses its FP32
weights for GEMV and the permitted BF16 hi/lo decomposition for GEMM.
"""

# v1: vectorized GEMV for one-token steps; pipelined BF16 Tensor Core GEMM with K-splitting for larger batches; router uses FP32 GEMV / BF16 hi-lo GEMM. Result: 4,337.4 µs (1.11x speedup).

from functools import lru_cache
from typing import TypedDict

import torch
import tilelang as tl
import tilelang.language as T


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


# Use an explicit pipeline without producer/consumer warp specialization.
_PASSES = {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}


@tilelang.jit(out_idx=[2], pass_configs=_PASSES)
def _gemv(n: int, k: int, weight_dtype: str):
    # Four warps/CTA, one output per warp for the wide BF16 projections.
    # The small router instead assigns a whole CTA to each output.
    bn = 1 if n <= 512 else 4
    bk = 512
    lanes = 128 // bn
    vec = 4 if weight_dtype == "float32" else 8

    @T.prim_func
    def main(
        X: T.Tensor((1, k), "bfloat16"),
        W: T.Tensor((n, k), weight_dtype),
        Y: T.Tensor((1, n), weight_dtype),
    ):
        with T.Kernel(T.ceildiv(n, bn), threads=128) as bx:
            w = T.alloc_fragment((bn, bk), weight_dtype)
            acc = T.alloc_fragment((bn, bk), "float32")
            total = T.alloc_fragment((bn,), "float32")
            T.annotate_layout({
                acc: T.Fragment(
                    (bn, bk),
                    forward_thread_fn=lambda i, j: i * lanes + (j // vec) % lanes,
                    forward_index_fn=lambda i, j: (j // (lanes * vec)) * vec + j % vec,
                ),
            })
            T.clear(acc)
            for ko in T.serial(T.ceildiv(k, bk)):
                T.copy(W[bx * bn, ko * bk], w)
                for i, j in T.Parallel(bn, bk):
                    acc[i, j] += w[i, j].astype("float32") * T.if_then_else(
                        ko * bk + j < k, X[0, ko * bk + j].astype("float32"), 0.0
                    )
            T.reduce_sum(acc, total, dim=1)
            for i in T.Parallel(bn):
                if bx * bn + i < n:
                    Y[0, bx * bn + i] = total[i]

    return main


@tilelang.jit(out_idx=[2], pass_configs={"tl.disable_warp_specialized": True})
def _projection(m: int, n: int, k: int, bm: int, bn: int, bk: int, splits: int):
    # Partition in whole K tiles so the final partition is simply zero-padded.
    # M tiles are adjacent CTAs, allowing them to reuse the same weight tile.
    # Larger tiles use Hopper TMA; short-M tiles use vectorized async copies.
    tiles_per_split = (k + bk * splits - 1) // (bk * splits)
    out_dtype = "float32" if splits > 1 else "bfloat16"
    out_shape = (splits, m, n) if splits > 1 else (m, n)

    @T.prim_func
    def main(
        X: T.Tensor((m, k), "bfloat16"),
        W: T.Tensor((n, k), "bfloat16"),
        Y: T.Tensor(out_shape, out_dtype),
    ):
        with T.Kernel(T.ceildiv(m, bm), T.ceildiv(n, bn), splits, threads=128) as (by, bx, bz):
            xs = T.alloc_shared((bm, bk), "bfloat16")
            ws = T.alloc_shared((bn, bk), "bfloat16")
            acc = T.alloc_fragment((bm, bn), "float32")
            ys = T.alloc_shared((bm, bn), out_dtype)
            T.clear(acc)
            for ko in T.Pipelined(tiles_per_split, num_stages=3):
                T.copy(X[by * bm, (bz * tiles_per_split + ko) * bk], xs, disable_tma=bm < 64)
                T.copy(W[bx * bn, (bz * tiles_per_split + ko) * bk], ws, disable_tma=bm < 64)
                T.gemm(xs, ws, acc, transpose_B=True)
            T.copy(acc, ys)
            if splits > 1:
                T.copy(ys, Y[bz, by * bm, bx * bn])
            else:
                T.copy(ys, Y[by * bm, bx * bn])

    return main


@tilelang.jit(out_idx=[3], pass_configs=_PASSES)
def _router(m: int, n: int, k: int, bm: int, bn: int, bk: int, splits: int):
    tiles_per_split = (k + bk * splits - 1) // (bk * splits)

    @T.prim_func
    def main(
        X: T.Tensor((m, k), "bfloat16"),
        Hi: T.Tensor((n, k), "bfloat16"),
        Lo: T.Tensor((n, k), "bfloat16"),
        Partial: T.Tensor((splits, m, n), "float32"),
    ):
        with T.Kernel(T.ceildiv(m, bm), T.ceildiv(n, bn), splits, threads=128) as (by, bx, bz):
            xs = T.alloc_shared((bm, bk), "bfloat16")
            hs = T.alloc_shared((bn, bk), "bfloat16")
            ls = T.alloc_shared((bn, bk), "bfloat16")
            hi_acc = T.alloc_fragment((bm, bn), "float32")
            lo_acc = T.alloc_fragment((bm, bn), "float32")
            ys = T.alloc_shared((bm, bn), "float32")
            T.clear(hi_acc)
            T.clear(lo_acc)
            for ko in T.Pipelined(tiles_per_split, num_stages=3):
                T.copy(X[by * bm, (bz * tiles_per_split + ko) * bk], xs)
                T.copy(Hi[bx * bn, (bz * tiles_per_split + ko) * bk], hs)
                T.copy(Lo[bx * bn, (bz * tiles_per_split + ko) * bk], ls)
                T.gemm(xs, hs, hi_acc, transpose_B=True)
                T.gemm(xs, ls, lo_acc, transpose_B=True)
            # Separate accumulators preserve the small correction terms.
            for i, j in T.Parallel(bm, bn):
                hi_acc[i, j] += lo_acc[i, j]
            T.copy(hi_acc, ys)
            T.copy(ys, Partial[bz, by * bm, bx * bn])

    return main


@tilelang.jit(out_idx=[1])
def _finish(m: int, n: int, splits: int, out_dtype: str):
    block = 256

    @T.prim_func
    def main(
        Partial: T.Tensor((splits, m, n), "float32"),
        Y: T.Tensor((m, n), out_dtype),
    ):
        with T.Kernel(T.ceildiv(m * n, block), threads=128) as bx:
            acc = T.alloc_fragment((block,), "float32")
            T.clear(acc)
            for s in T.serial(splits):
                for i in T.Parallel(block):
                    if bx * block + i < m * n:
                        acc[i] += Partial[s, (bx * block + i) // n, (bx * block + i) % n]
            for i in T.Parallel(block):
                if bx * block + i < m * n:
                    Y[(bx * block + i) // n, (bx * block + i) % n] = acc[i]

    return main


@lru_cache(maxsize=None)
def _plan(m: int, n: int, k: int, router: bool):
    """Shape-only launch heuristic; cached values contain no input/output data."""
    bm = 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128
    if router:
        bn, bk = 32, 64
    elif m <= 32:
        bn, bk = 32, 128
    else:
        bm, bn, bk = 64, 64, 64
    blocks = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
    splits = 1
    # Populate the 132-SM H200 without excessive partial-output traffic.
    # K slices are at least 256 elements for BF16 and 128 for the router.
    minimum_k = 128 if router else 256
    while blocks * splits < 240 and splits * 2 * minimum_k <= k:
        splits *= 2
    if router:
        return _router(m, n, k, bm, bn, bk, splits), _finish(m, n, splits, "float32")
    return _projection(m, n, k, bm, bn, bk, splits), (
        _finish(m, n, splits, "bfloat16") if splits > 1 else None
    )


def prepare(weights: Weights) -> dict[str, torch.Tensor]:
    """Prepare only immutable weights; every activation is handled in kernel_fn."""
    state = {name: weight.contiguous() for name, weight in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    state["router_hi"] = hi
    state["router_lo"] = (router - hi.float()).to(torch.bfloat16)
    return state


def kernel_fn(data: Inputs, state: dict[str, torch.Tensor]) -> torch.Tensor:
    x = data["x"]
    config = data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]
    router = layer == "router"
    if m == 1:
        return _gemv(n, k, "float32" if router else "bfloat16")(x, state[layer][:n])
    compute, finish = _plan(m, n, k, router)
    if router:
        partial = compute(x, state["router_hi"][:n], state["router_lo"][:n])
    else:
        partial = compute(x, state[layer][:n])
    return finish(partial) if finish is not None else partial
