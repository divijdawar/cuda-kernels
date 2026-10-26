# TileLang kernels for OpenKernelChallenge

This directory collects TileLang kernels developed for OpenKernelChallenge problems. Each problem has its own subdirectory so implementations, notes, and results stay together.

## OpenKernelChallenge problems

The [OpenKernelChallenge](https://openkernelchallenge.dev/) asks participants to speed up Muse Spark Open's GPU kernels against SGLang on H200 GPUs. The list below summarizes all five kernel problems; this directory currently contains an implementation only for `muse_bf16_proj`.

- [`muse_bf16_proj`](https://openkernelchallenge.dev/problems/muse_bf16_proj) — Dense attention and MoE projections plus the router, computing `x @ W.T`; projection weights are BF16 and router weights are FP32.
- [`muse_attention_decode`](https://openkernelchallenge.dev/problems/muse_attention_decode) — Gated grouped-query attention for decode or 8-token verification, attending over paged FP8 KV cache with sliding-window or full-context attention.
- [`muse_moe_w4a16`](https://openkernelchallenge.dev/problems/muse_moe_w4a16) — Routed top-8 of 256 experts, with SwiGLU, NVFP4 weights, per-expert output normalization, weighting, and accumulation.
- [`muse_decode_layer`](https://openkernelchallenge.dev/problems/muse_decode_layer) — A complete Muse decoder layer on one GPU, combining projections, attention, routing, expert computation, normalization, and residual operations.
- [`muse_prefill_layer`](https://openkernelchallenge.dev/problems/muse_prefill_layer) — A complete decoder layer for long prefill workloads; attention and routed experts are included, with grouped NVFP4 expert GEMM as the main workload.

## `muse_bf16_proj` details

[`muse_bf16_proj`](./muse_bf16_proj/) contains the TileLang implementation for dense projections. The workload is `out = x @ W.T` on NVIDIA H200 GPUs, with `x` shaped `[m, k]`, weights shaped `[n, k]`, and output shaped `[m, n]`.

- `qkv_gate`, `o`, `pooler`, and `unpooler` use BF16 weights and produce BF16 outputs.
- `router` uses FP32 weights and produces FP32 outputs. The rules permit a BF16 high/low weight decomposition with FP32 accumulation for GEMM workloads.
- The scored batch sizes (`m`) include 1, 8, 16, 32, 128, and 256. Additional shapes are used for correctness checks.
- Correctness is checked against an FP64 reference with relative-L2 error limits: 1% for regular projections and 0.01% for router outputs. Extra checks cover individual rows and outlier distributions.
- Timing uses CUDA graph replays with cold L2, compared against SGLang on the same GPU. The problem score is a geometric mean across the defined loads.
- Submissions must compute from the current inputs and supplied weights, obey the specified precision, and avoid host synchronization or timing/harness manipulation inside timed calls.

See the [problem page](https://openkernelchallenge.dev/problems/muse_bf16_proj) and [competition rules](https://openkernelchallenge.dev/rules) for the authoritative specification.

