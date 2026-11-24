# Llama-3.1-8B-Instruct engine on Trainium2 (one logical NeuronCore, LNC=2): notes

## Layout of the source

- `build.sh`: compiles every NKI kernel variant to a NEFF (`kernels/build_neffs.py`, NKI's documented route: compile
  and one test run on the experiment core, `ENGINESMITH_NEURON_CORE + 2`), caches them under
  `target/neffcache/` keyed by the kernel module + `common.py` + `build_neffs.py` (+ `kernels/nkattn/` for prefill),
  embeds them with `.incbin` (`target/gen/`), links `host/main.cpp` against libnrt only. A clean build takes ~9 min
  with the default 6 jobs (`NEFF_BUILD_JOBS`; the 12 `dec_grp16_*` NEFFs ~160 s each, `pf_layer_4096` ~200 s).
- `kernels/common.py`: constants, column-layout GEMV, RMSNorm, cross-core exchange.
- (round 011) The prefill of a whole prompt in the 256 / 1024 buckets is 4 executions: `pf_first_256` (embed + layer 0;
  1024: `pf_embed` + `pf_layer_1024`), 2 x `pf_multi15_S` (layers 1-15 / 16-30 in one 15-layer NEFF, host tensor sets
  `F.multi_in`), `pf_lasth_S` (last layer + final norm + LM head + argmax, writes toks[pos+1]). 2048 / 4096 keep the
  per-layer NEFFs but end in `pf_lasth_S`. S = 256 layers exchange bf16(h1 * g2) (`NS_XN2X`) and accumulate 8 output
  tiles per own-K-half batch (`NS_NB`, `NS_NBD`). Build keys no longer include build_neffs.py itself. See Round 011.
- (round 009) `pattn.attention_t`: transposed causal prefill attention (S^T with K stationary, no row max, no P
  transposes) used for S >= 2048 (`PF_ATTN_T_MIN` in prefill.py); `attention_heads` stays for S <= 1024. Phase C of
  `pf_layer_tok` / `pf_layer_pipe` sums the MLP-norm squares per o_proj output tile (`PF_TOK_SQ_STREAM`,
  `PF_PIPE_SQ_STREAM`, `_norm_fm_from_sum`). See Round 009.
- (round 008) `pf_layer_ns` (S <= 256) now also takes `xa` (fp32 scratch for the partial sums of squares; the host binds
  whichever of xa/xb/hs/ms a layer NEFF takes); its exchanges are restructured (`NS_OWNF`, `NS_PSX`, `NS_POSTR`, see
  Round 008). The last layer at S >= 1024 computes Q / attention only for the 128 queries ending at pos (`LAST_Q`,
  `_last_qattn` in prefill.py).
- (round 006) Decode MLP split is asymmetric: `MLP_SHIFT = 4` in decode.py (core 1 also computes core 0's last 4
  intermediate tiles; see Round 006). `gemv_ring_rows` (decode.py): ring GEMV over a column range of a strided run.
- (round 005) The decode step is 2 executions: `dec_grp16e_B` (embed + layers 0-15) and `dec_grp16h_B` (layers 16-31
  + final norm + LM head + argmax); `dec_grp16_B` is no longer built (`GRP_FUSE` in build_neffs.py). `dec_embed`,
  `dec_head` and `dec_layer_B` remain (prefill head, `ENGINE_NO_FUSE=1` fallback).
- `kernels/decode.py`: `dec_embed` (gather `embed[toks[pos]]`), `dec_layer` (one transformer layer for one token,
  static attention bucket `B`; used only with `ENGINE_NO_GRP=1`), `dec_grp16` (16 consecutive layers in one NEFF,
  the decode path since round 002; the 128-argument signature is spelled out because NKI parses the source),
  `dec_head` (final norm, LM head, argmax with ties to the lowest id, writes
  `toks[pos+1]` and returns `pos+1`).
- `kernels/pattn.py` (round 004): causal prefill attention in per-group stages, bitwise equal to the vendored nkilib
  attention_cte; `attention_heads` is the drop-in.
- `kernels/prefill.py`: `pf_embed`, three layer variants over a prompt bucket of S tokens -- `pf_layer_ns` (S <= 256:
  QKV/attention by heads, o_proj/down split along N with bf16 activation exchanges), `pf_layer` (S = 512: heads /
  intermediate tiles, fp32 partial exchanges) and `pf_layer_tok` (split by tokens, S >= 1024; `PF_NS_MAX`,
  `PF_NSPLIT_MAX` in build_neffs.py) -- and `pf_pick` (last token's hidden into the decode head's column layout).
  Attention in all of them: `_attention` = nkilib `_attention_cte` (vendored in `kernels/nkattn/`, THIRD_PARTY.md).
- `host/main.cpp`: weight loading and re-layout (values unchanged), device tensors, tensor sets, queued
  executions (`nrta_execute_schedule`), the JSON-lines protocol. `host/json.h`: minimal JSON parser.

## Design (round 001; round-002 changes marked)

- Weight tiles: W [N, K] (HF row = output) is stored as 128x128 tiles `tile[p, j] = W[nt*128+j, kt*128+p]`, one
  contiguous run per physical core `[128, NT*KT*128]` (tile (nt,kt) at column (nt*KT+kt)*128). The tile is the
  stationary operand of both the decode GEMV (moving = one activation column) and the prefill GEMM (moving = 256
  tokens of a feature-major activation). One copy of the weights serves both (16.1 GB on the device).
- Decode split over the 2 physical cores: QKV by heads (q 16c..16c+15, kv 4c..4c+3), o_proj K-split over the core's
  heads, gate/up N-split over intermediate tiles 56c..56c+55, down K-split over the same tiles; fp32 partials summed
  with `nisa.sendrecv` (gpsimd DMA). LM head split over the vocabulary (501 tiles per core).
- Decode step = 4 queued executions since round 002: `dec_embed`, 2 x `dec_grp16_B` (layers 0-15, 16-31; per-group
  tensor sets binding the per-layer tensors as `g1_0 .. vc_15`), `dec_head`. (Round 001: 34, one NEFF per layer.)
  Round-001 text: `dec_head`. The
  position lives in a device tensor; two tensor-set parities swap `P0`/`P1`; tokens stay in a device `toks` array;
  the host reads tokens once per request. Buckets B = 384, 512, 768, 1024, 1152, 1536, 2048, 2560, 3072, 3584,
  4096, 4224 (smallest B >= pos+1).
- KV cache per layer: K transposed `[8, 128, 4224]`, V `[4224, 1024]`, bf16. The decode layer writes this token's
  K/V at `pos`, then loads the cache [0, B) (same tensors, so the compiler orders the loads after the writes) and
  masks keys > pos with -30000.
- Prefill: bucket S in {128, 256, 512, 1024, 2048, 4096} (smallest S >= prompt), prompt padded (pad K/V written
  beyond the prompt are overwritten by decode before they are read). Activations feature-major in HBM
  `[128, 32, S]`. Causal attention per (head, 128-query block): scores in SBUF, P^T by identity matmuls, PV with V
  stationary giving O^T feature-major.
  - `pf_layer` (S <= 512): QKV by heads per core, attention on own heads (no barrier), o_proj and down K-split with
    fp32 partials exchanged through HBM scratch (`xa`, `xb` [2,128,32,S]) + `core_barrier`, gate/up N-split;
    every weight tile read by one core per 512-token chunk; each core stores half of the output features.
  - Round 002: attention replaced by nkilib `attention_cte` (below); `pf_layer_ns` for S <= 256 (below).
  - `pf_layer_tok` (S >= 1024): core c owns tokens [c*S/2,(c+1)*S/2) for the GEMMs (256-token chunks, each core
    streams the whole layer's weights), barriers on kc/vc/qs, attention on interleaved query blocks
    (core c: blocks c, c+2, ...: balanced causal work), barrier on att, then o_proj/MLP per chunk. Prompts > 4096: prefill 4096, then decode steps (heads write to a junk token buffer except
  the last).
- Numerics follow NxD Inference's rounding points: RMSNorm bf16(x * rstd * g) in fp32; q/k/v rounded to bf16; RoPE
  with bf16 cos/sin tables and bf16 products/sum; scores bf16(bf16(q.k) * 1/sqrt(128)); softmax fp32, P bf16;
  attention output bf16; residual adds in bf16; logits rounded to bf16, argmax with ties to the lowest id.
- **Reference quirk reproduced (1-token prompts):** NxD's context encoding pads `position_ids` with 1 and takes the
  output row at `argmax(position_ids)`, so for a one-token prompt its first token is the prediction after
  `[prompt, 0]` at position 1 (pad_token_id None -> 0, right padding, causal mask). Measured: with the exact model
  the first token of a 1-token prompt mismatched with rank 109,521 (margin 13.5); the engine now prefills
  `[t, 0]` and continues from position 1 like the reference (control-p1 identical).

## Verification

- Kernel unit tests against numpy (lab, real layer-0 weights): decode layer at B=512/1152/4224 and pos 0/300/1100/4150:
  max diff 1 bf16 ulp in 0.02-0.7% of outputs (fp32 summation order). Prefill layer S=128/256/4096: rel. L2 ~1e-3
  (1-ulp differences in ~10% of outputs: different reduction order of the norm). Argmax tie rule checked with
  injected logits (ties within a partition, across partitions, across cores).
- dev-eval 002 (1 round) and 004 (3 rounds): PASSED; every mismatch is the reference's rank-2 token.

## Prefill changes after dev-eval 002 (end-to-end prefill request, 4 fresh prompts each, lab/drive.py, core 50)

| change | 256 | 1024 | 4096 |
|---|---|---|---|
| dev-eval 002 version (token split, contiguous halves) | 49.3 ms | 122.7 | 767 |
| interleaved query blocks between cores + fewer softmax passes | 49.3 | 122.7 | 636.6 |
| head/intermediate split for all S (`pf_layer`, MLP chunk 512) | 35.6 | 129.0 | 672.9 |
| hybrid: `pf_layer` for S<=512, `pf_layer_tok` for S>=1024 (current) | 35.5 | 122.7 | 636.6 |

Rejected (neutral or worse): 512-token chunks for the token-split QKV phase (profile 20.27 vs 19.88 ms per layer);
2 MB weight slabs in the decode GEMVs (decode layer profile 0.760 vs 0.759 ms). Note: neuron-explorer `total_time`
of an LNC=2 NEFF did not match end-to-end times when the two cores had unequal work (old token split: profile
15.9 ms/layer, real ~24 ms): compare variants end to end.

Decode profile (dec_layer_384): 0.759 ms device time, 438 MB read (577 GB/s, MBU 0.80, DMA active 98%); dec_head
1.70 ms, 1.05 GB. 32 layers + head ~ 26 ms per step in the profile, ~25.2 ms measured. The remaining gap to
~650-700 GB/s is probably per-NEFF bubbles (34 executions per step, no weight prefetch across layers): a
whole-step NEFF (all layers in one, stacked weight tensors, prefill binding per-layer slices with
`nrt_tensor_allocate_slice`) is the next decode lever; not tried.

## Measurements (dev-eval 002, 1 round, before the prefill changes)

| case | prefill ms ref / engine | decode (64 tok) ms ref / engine |
|---|---|---|
| 256 | 69.1 / 49.3 (1.40x) | 1874.9 / 1610.6 (1.16x) |
| 1024 | 241.8 / 124.3 (1.95x) | 2094.4 / 1620.6 (1.29x) |
| 4096 | 606.5 / 767.4 (0.79x) | 2010.1 / 1638.4 (1.23x) |

score 1.2586 (prefill 1.291, decode 1.227). Decode ~25.2 ms/token; floor estimate 14.4 GB / ~650 GB/s = 22 ms.

## Round 001 engine: dev-eval 004 (3 rounds, public prompts): PASSED, score 1.3765 (prefill 1.537, decode 1.232)

| case | prefill ms ref / engine | decode (64 tok) ms ref / engine |
|---|---|---|
| 256 | 68.6 / 35.5 (1.93x) | 1885 / 1609 (1.17x) |
| 1024 | 242.2 / 122.8 (1.97x) | 2104 / 1620 (1.30x) |
| 4096 | 606.6 / 636.9 (0.95x) | 2013 / 1636 (1.23x) |

Long-output check: 200-token prompt + 3840 tokens, a 4000-token prompt with 224 tokens in two requests, and a
4097-token prompt + 65 tokens all return every token (lab, before the prefill changes; the decode path is unchanged).

## Round 002: what changed, measured

All A/B numbers: `brief/skill/scripts/ab.py`, both engines loaded on spare cores 50/51 (wrappers set
`NEURON_RT_VISIBLE_CORES`), fresh nonce prompts, split request (prefill = generate(P,1), decode = generate((),64)),
3-6 rounds, medians. Profiles: `neuron-explorer capture` of one NEFF on core 50, `total_time` of the 3rd execution.

| change | measured | tokens |
|---|---|---|
| prefill attention = nkilib `attention_cte` (as NxD: q pre-scaled by bf16(q/sqrt(128)), scale 1, causal, tp_q/tp_k False, tp_out True), K/V also written to contiguous `kq [8,128,S]`, `vq [8,S,128]` scratch, output `att [32,128,S]` | prefill 4096 636.8 -> 477.9 ms, 1024 123.2 -> 116.4, 256 unchanged; dev-eval 001: PASSED, 0 mismatches, score 1.4586 | differ from round 001 (closer to the reference: 0 mismatching steps) |
| decode: 4 layers per NEFF (`dec_grp4`) | profile 0.760 -> 0.730 ms per layer (b384); decode 1609.6 -> 1544.5 ms (+4.2%) | identical |
| decode: 16 layers per NEFF (`dec_grp16`, all 12 buckets) | profile 0.7288 (G=8), 0.7252 (G=16) ms per layer; decode +4.7 / +5.0 / +3.0% at 256 / 1024 / 4096 vs round 001 kernels | identical |
| KV-cache loads on hardware DGE (Sync queue) instead of software DGE | grp4 b384 profile 2.920 -> 2.899 ms; decode +1.0 / +0.4 / +0.6% | identical |
| `pf_layer_ns` for S <= 256 (o_proj/down N-split reading both per-core runs of the decode layout, bf16 exchanges of att, h1, MLP intermediate instead of fp32 partials) | profile 1.0517 -> 1.0283 ms per layer; prefill 256 36.0 -> 35.0 ms (+2.7%) | identical on the ab prompts |

dev-eval 003 (1 round, the final kernels before cleanup): PASSED, score 1.4929 (prefill 1.7324, decode 1.2866), 0
mismatching steps:

| case | prefill ms ref / engine | decode (64 tok) ms ref / engine |
|---|---|---|
| 256 | 68.5 / 35.1 (1.954x) | 1879 / 1526 (1.231x) |
| 1024 | 241.9 / 115.8 (2.089x) | 2094 / 1541 (1.359x) |
| 4096 | 606.4 / 476.0 (1.274x) | 2009 / 1579 (1.272x) |

Final source (after cleanup, built by dev-eval): dev-eval 005, 3 rounds: PASSED, score 1.4917 (prefill 1.7320,
decode 1.2847), 0 mismatching steps in every round; prefill 34.9 / 115.7 / 476.0 ms, decode 1525.6 / 1540.3 /
1578.5 ms at 256 / 1024 / 4096.

Verification: same.py (40 prompts x 3 request shapes, 1..4097 tokens) between the dev-eval-003 binary and a clean
build of the final source: 40/40 identical, shapes self-consistent. Long-output check (lab script): 200-token prompt +
3840 tokens, 4000 + 161, 4097 + 65, 3000 + 1100, 4161 + 1, 1/2/129/513-token prompts, and a continuation to position
4161: every request returned all its tokens.

### Where the time is now (measured), and why the remaining levers are small

- Decode ~23.8 ms/token (15.0 GB per step, ~630 GB/s). In-NEFF, the weight stream runs at MBU 0.84 with DMA 99%
  busy. Micro-benchmark (`stream` kernel, 872 MB HBM->SBUF, hwdge Sync): 4 KB per partition per DMA 588 GB/s,
  8 KB 617, 16 KB 647, 32 KB 657 GB/s without matmuls; with one 1-column matmul per 128 columns (as the GEMV)
  617-631 GB/s. The TensorE itself is not the limit (resident 128x128 stationary + 1-column moving: 38.6 ns per
  matmul = 849 GB/s per physical core). So the GEMV-plus-stream is at ~95% of the measured stream ceiling.
- Prefill 1024 / 4096: TensorE active 94% / 96.6% (throttle limit ~0.91-0.95). TensorE rate measured resident:
  0.43 ns per moving column (128x128 stationary), no meaningful fixed cost (N=128: 59.5 ns, 256: 112.8, 512: 219),
  so wider moving tiles would not help. The linear layers run at ~136-148 TFLOP/s while active.
- Prefill 256: TensorE 81% active, throttle limit 0.85; 1.028 ms per layer.

### Tried and rejected this round (numbers)

- Weight DMAs alternating Sync / Scalar hardware DGE queues (decode, grp4 b384): 2.947 vs 2.920 ms (worse).
  Tensor-engine queue: not allowed for hwdge (compiler assertion).
- Larger weight DMAs in decode (16 / 24 / 32 KB per partition): 2.915 / 2.919 / 2.912 vs 2.920 ms (no change,
  although the pure stream gains from them).
- KV-cache loads with `dge_mode.none` at b4224: 3.0095 vs 3.0107 (sw) vs 3.0088 (hw) ms: no difference at b4224.

### Next ideas (not measured)

- Whole decode step in one NEFF (embed + 32 layers + head): G=4 -> 16 bought only 0.7% device time; expect <=0.5%.
- nkilib attention: P transposes on the TensorE; at 4096 the TensorE is the bottleneck, so DMA transposes
  (`nisa.dma_transpose`) could buy a few % of the attention (attention ~10% of prefill PE work).
- Prefill 256: 19% TensorE idle (phase boundaries / barriers); several layers per NEFF could overlap the next
  layer's first weight loads.

## Round 003: what changed, measured

Tools (session dir `lab/`, not part of the engine): `A.sh`/`B.sh` run `lab/bin/{A,B}` on cores 50/51
(`NEURON_RT_VISIBLE_CORES`), `GPU_LOCK_FILE=lab/gpu.lock`; `one.sh NAME` builds one NEFF variant on core 51;
`prof.sh NEFF TAG` = neuron-explorer capture on core 50 (+ `prof/ana.py` per-engine busy and DMA GB/s per time bin,
`prof/cores.py` per-core DMA bytes, `prof/ops.py` per-opcode time); `attn_test.py` runs `_attention_cte` alone on
random q/k/v (S = 256/1024/4096) and saves the output, so a kernel change is checked bitwise against the stock copy
(`lab/at_orig/`) and timed in isolation. **Core bias (A/A, same binary):** core 51 is 0.1-0.5% slower than core 50 for
prefill and 0.2-0.35% for decode; the A/B numbers below are raw (B on core 51) unless marked corrected.
**Profiles are noisy now:** other runs share the machine; single captures came out 1.5-3x slow at times (throttled);
repeat every capture 2-3 times and take the consistent value.

| change | measured | tokens |
|---|---|---|
| last layer (`pf_last_S`, S <= 256 and S >= 1024): QKV + attention for all tokens (K/V cache), then o_proj/MLP for the last prompt token only (the `_pf_cde_nsplit` arithmetic on one column: matmul columns are independent), writing the decode head's column `h` directly (no `pf_pick`) | prefill +0.9 / +1.9 / +2.3% (256 / 1024 / 4096), ab 4 rounds | identical (same.py 40/40) |
| nkilib attention: causal diagonal 512-key tiles masked by one `range_select` from PSUM (mask + max) instead of tensor_copy + affine_select + tensor_scalar_reduce (scale is 1.0: same values) | attention alone 1024: 468 -> 406 us, 256: 115 -> 107 us | bitwise vs stock kernel |
| nkilib attention: Q/K/V tile loads on hardware DGE (Sync queue) instead of GpSimd software DGE (the O store does not accept HWDGE) | 1024: 406 -> 385 us, 256: 107 -> 95 us | bitwise |
| nkilib attention: QK matmul, mask and exp of a diagonal tile stop at key 128*(group+1) (all later keys are masked: exp = 0, adds nothing to the row sum, does not change the max) | 1024: 385 -> 329 us, 256: 95 -> 90 us, 4096: 2478 -> 2443 us (stock 2572) | bitwise |
| S <= 256 layer: own halves of h1 / MLP intermediate stay in SBUF (only the other core's half read back), exchanges stored in pieces, norm squaring split in 4 | profile 1.033 -> 1.024 ms per layer | identical |
| S <= 256 layer: x (layer input), attention output and the other core's h1 / MLP halves loaded on the Sync queue (ahead of the queued weights; they were sharing bandwidth with the weight prefetch at ~70-130 GB/s); other exchanges on the Scalar queue | 1.0 -> 0.97 -> 0.961 ms | identical |
| S <= 256 down projection: weight slabs (56 tiles, 14 KB/partition) through an explicit 8-slot SBUF ring | 1.004 -> 0.976 ms (5 slots: 0.981) | identical |
| S >= 1024: phase-A x chunks on the Sync queue + split norm | profile 3.47 -> 3.45 ms (1024) | identical |
| decode: GEMV weights through an explicit ring (`gemv_ring`, `RING_NB` slots shared by all GEMVs of a 16-layer NEFF, so DMAs run ahead across GEMV and layer boundaries); 16 KB slots (6) for buckets <= 2048, 8 KB (10) above | decode +0.2 / +0.3 / +0.45% (8 KB ring, corrected), 16 KB slots another +0.2 / +0.75 / +0.1% | identical |
| decode head: LM-head weights through a 4 x 16 KB ring | dec_head profile 1.691 -> 1.647 ms | identical |

Result: dev-eval 001 (1 round, built by dev-eval): **PASSED, score 1.5350 (prefill 1.8205, decode 1.2943)**, 0
mismatching steps; prefill 33.0 / 109.7 / 458.8 ms, decode 1518.5 / 1524.2 / 1571.5 ms at 256 / 1024 / 4096
(round 002: 34.8 / 115.6 / 475.6 and 1525 / 1540 / 1578). Final source (after cleanup), dev-eval 002, 3 rounds: **PASSED, score 1.5337 (prefill
1.8169, decode 1.2947)**, 0 mismatching steps in every round; prefill 33.2 / 109.6 / 459.4 ms, decode 1518.2 /
1523.8 / 1571.3 ms. Long-output check (`lab/longcheck.py`, vs the round-002 binary): 200 + 3840 tokens, 4000 + 161,
3000 + 1 + 1099, 4161 + 1, 4097 + 65, 1/2/129/513-token prompts, 2049 + 20 + 2000: all identical, all tokens returned. same.py vs the round-002 binary: 40/40 identical, shapes
self-consistent (also after the final cleanup, which only removed switched-off experiments).

### Where the time is now (measured)

- Prefill 1024 / 4096: the GEMMs run at the TensorE peak (0.43 ns per moving column, ~2.93 ms of linear work per
  layer per 1024 tokens); per layer at 1024: 3.43 ms = GEMM 2.93 + attention 0.33 + 0.17 other; at 4096: 14.35 ms =
  GEMM ~11.7 + attention 2.44 (Scalar engine ~80-90% busy: one exp + one `ACTIVATION_READ_ACCUMULATOR` (~280 ns) per
  512-key tile) + ~0.2.
- Prefill 256: 0.96 ms per layer vs ~0.72 GEMM/stream floor: preamble ~15 us, x load + norm ~20 us, attention
  ~90 us (latency-bound, no engine saturated), two exchange gaps (~40-50 us each), down projection DMA-bound.
- Decode: 32 layers at ~635 GB/s (the measured stream ceiling for 8 KB DMAs with matmuls is 617-631, 16 KB 647).
  In profiles core 0 streams slower than core 1 and core 1 waits ~50-70 us per layer at an exchange; that may be the
  profiler (trace DMA on core 0): the end-to-end layer time is ~4% below the profiled one.

### Tried and rejected this round (numbers)

- nkilib attention, all DMAs with static descriptors (`dge_mode.none`): neutral in isolation, end to end prefill
  -1.0 / -0.4 / -0.5%. HWDGE for every DMA does not compile (`Unsupported DGE Type`, the O store and some SBUF copies).
- nkilib attention, exp row sums chained in the Scalar engine's reduce registers over a group's tiles (one read-out
  per group instead of per tile): 4096 attention 2443 -> 2227 us, but **not bitwise** (57 of 4.2M outputs differ at
  1024, 1270 of 16.8M at 4096: the Vector partial-sum reduce is not left to right). Not used.
- nkilib attention, multiply by the transposed reciprocal straight from PSUM (drop a copy): 1024 329 -> 407 us; the
  copy on the Scalar engine: 1024 -> 402 us, 256 90 -> 104 us. Engine assignment changes upset its scheduling.
- Stacking the 4 q heads of a KV group into one 4S sequence (sequence-packing bounds; deeper group pipeline, one K/V
  load): bitwise identical, 256: 94 -> 103 us. Rejected.
- Two prefill layers per NEFF (S = 256): 1.954-1.971 ms vs 2 x 0.960 ms. Rejected.
- o_proj ring with its first slabs prefetched during attention (S = 256): neutral (0.981 -> 0.984 ms); QKV ring:
  neutral.
- Decode KV-cache loads on the Scalar queue (to keep the Sync weight queue from blocking on them): decode -0.6%
  (corrected). 16 KB decode ring slots at bucket 4224: -0.85% (SBUF pressure next to the KV tiles), hence 8 KB there.

### Next ideas (not measured)

- 4096 attention is Scalar-engine bound by the per-512-tile exp + read-out pairs; only a non-bitwise change (chained
  sums, above) was found. A bitwise variant would need the Vector partial-sum order reproduced.
- S = 256: attention (~9% of the layer) and the two exchange gaps; a hand-written small-S attention that reproduces
  nkilib's instruction sequence per (head, group) but pipelines across heads.
- Decode: fusing `dec_embed` into the first and `dec_head` into the second group NEFF (2 instead of 4 executions per
  step, ~0.2%).

## Round 004: what changed, measured

Tools (session dir `lab/`, not part of the engine): `A.sh`/`B.sh` (round-003 binary / candidate on cores 50/51),
`one.sh NAME [OUT]` builds one NEFF variant on core 51 (~20 s at S=256, ~50 s at 1024, ~180 s at 4096),
`stream/prof.sh NEFF` = neuron-explorer `total_time` (repeat 2-3x: the first capture of a new NEFF is sometimes
1.5-2x slow), `prof/cap.sh` + `prof/act.py` (**per physical core** engine busy per time bin from the profile's
`active_time` records; the `instruction` list merges both cores) + `prof/bins.py` (matmul / ACT / table-load counts per
bin, opcode totals), `attn/` (attention kernel alone on random q/k/v, compared bitwise with the vendored nkilib).

| change | measured | tokens |
|---|---|---|
| `kernels/pattn.py`: own causal prefill attention, per (head, 128-query group) stages (`stage_qk`, `stage_exp`, `stage_tp`, `stage_pv`, `stage_wb`) that issue exactly nkilib attention_cte's instructions for our config (MM1 per 512-key tile, range_select / tensor_scalar_reduce + per-tile max, negated max reduce, activation_reduce exp + per-tile sums, P^T, MM2 per 2048-key PSUM tile + fp32 add, sum reduce, reciprocal, nc_transpose of the reciprocal, multiply). `attention_heads` (heads split as nkilib, groups pipelined QK g / exp g-1 / PV g-2) replaces `_attention_cte` everywhere (`PF_ATTN`) | attention alone vs nkilib: 4096 2.453 -> 2.207 ms, 1024 0.325 -> 0.316, 256 0.092 -> 0.090; layer profiles 256 0.969 -> 0.930, 1024 3.441 -> 3.409, 4096 14.66 -> 14.31 ms; ab 4 rounds prefill +3.7 / +1.2 / +2.5% | bitwise vs nkilib (0 of 4.2M / 16.8M / 1M outputs differ); same.py 40/40 |
| P^T on the TensorE (`nc_transpose` + one PSUM->SBUF copy per key tile, `pe_tp`) instead of DMA transposes; output rounded to bf16 by the engine into SBUF instead of by the fp32->bf16 DMA store | used only in the pipelined layers below (alone at 4096 it is slower: 2.87 vs 2.21 ms) | bitwise vs nkilib (both checked on 21M outputs) |
| S <= 256 (`_ns_phase_ab`, `NS_PIPE`): QKV in kv-group order (k_g, v_g, q 4g..4g+3); the 8 attention units of group g read q/k/v from SBUF (no qs/kq/vq scratch) and are emitted between the QKV GEMMs of group g+1 (QK u, exp u+1, P^T u+2 on PE, PV u+3, store u+4) | layer 0.930 -> 0.89 ms; ab prefill 256 +8.0% vs round 003 | same.py 40/40 |
| S = 1024 (`pf_layer_pipe`, `PF_PIPE_S`): each core owns 256-token chunks (core 0: 0,3; core 1: 1,2: equal causal work), phase A on own chunks, K/V barrier, attention by query chunk (all 32 heads) interleaved with the previous chunk's o_proj / down steps (not gate/up, see table loads below), attention output kept in SBUF as the o_proj input | layer 3.409 -> 3.31 ms; ab prefill 1024 +2.3% vs round 003 | same.py 40/40 |

Result (ab, 4 rounds, vs round 003, B on core 51): prefill 33.3 -> 30.9 / 109.7 -> 107.2 / 459.6 -> 449.9 ms
(+8.0 / +2.3 / +2.1%); decode unchanged (-0.2%, the core-51 bias; decode NEFFs are the round-003 source).
dev-eval 001 (2 rounds): PASSED, score 1.5654. Final source (diagnostic switches removed, clean build): same.py 40/40
vs round 003, long-output / bucket-edge check (`lab/longcheck.py`: 200+3840, 4000+161, 4097+65, 4161+1, prompts
1/2/129/255/256/257/513/1000/1023/1024/1025/2049/3333) all tokens returned and identical to round 003; **dev-eval 002
(3 rounds, --no-build): PASSED, score 1.5721 (prefill 1.9000, decode 1.3008)**, 0 mismatching steps; prefill
30.7 / 107.1 / 449.6 ms, decode 1518.9 / 1522.8 / 1570.3 ms at 256 / 1024 / 4096.

### Tried and rejected (numbers)

- Decode weight stream micro-benchmark (`lab/stream`, 268 MB HBM->SBUF with 1-column matmuls): row layout with a
  1 MiB partition stride 0.665 ms vs 0.453 ms for any other stride (128/256 KB, 917 KB: the engine's strides are
  fine); contiguous 2 MB blocks, 32 KB DMAs, 12-slot ring, alternating Sync/Scalar queues: all 0.453-0.455 ms. No
  decode bandwidth left to find in layout or queues.
- Pipelined layer at S = 4096 (same design as 1024): 15.4 ms first, then 15.1 (no exp next to SiLU), 15.6 (PE
  transposes), 15.8-17 ms with other distances / windows, vs 14.31 for the sequential layer. Findings:
  (1) exp and SiLU live in different activation tables: interleaving them cost 1478 ACT_TABLE_LOADs (1.3 us each)
  per layer -> attention steps only beside o_proj / down; (2) the in-order engines convoy: a TensorE op of the
  attention placed one step after its producer stalls all following GEMMs when the Vector / Scalar chain behind it is
  longer than the step (late chunks have 8 key tiles per group); (3) DMA transposes (software DGE on GpSimd) and the
  fp32->bf16 store DMA queued behind weight DMAs added ~1-2 ms. Baselines: phases A + C alone (no attention) 12.07 ms,
  sequential chunked attention 14.61, heads attention 14.31. A version that splits every stage into single-engine
  sub-stages one step apart might reach ~13 ms but needs more SBUF than the MLP leaves at 4096; not done.
  Diagnostic variants that drop the PV or store stage are not valid lower bounds: the compiler removes the unused
  attention work.
- QK PSUM tiles cycling through 2 buffers (`QK_PS`): no change. S=256 pipeline knobs (distances 1/1/2/3 and
  1/2/4/6, ramp over 3 tiles, DMA transposes): 0.90-0.91 vs 0.89 ms.

### Next ideas (not measured)

- Last prefill layer: Q and attention are computed for all tokens but only the last token's row is used (~2 ms at
  4096, ~0.5 ms at 1024). The group holding the last token is dynamic (pos on the device); computing one dynamic
  group per head with all key tiles (extra tiles fully masked: exp = 0) should give the same bytes but needs a check.
- S = 4096 pipelining with single-engine sub-stages (see above).

### Pitfalls found (round 004)

- `nl.program_id(0)` is a trace-time int per core: per-core control flow (chunk lists, loop counts) works.
- Plain Python classes do not trace in kernel code; lists (append, index, item assignment) and `while` on trace-time
  values do.
- `dma_transpose` with 128 source partitions cannot use hardware DGE (limit 16): it is software DGE on GpSimd.
- The DMA `priority` argument is NeuronCore-v4+ only.

## Pitfalls found (NKI 0.6, measured)

- (round 003) The NKI frontend parses kernel source: a generator expression (`sum(x for ...)`) fails with
  `unsupported expression`; plain loops, lists and `if` on trace-time ints work. Indentation slips are silent: an
  edit that moved the P transpose of nkilib's `_exp_impl` into an `else` branch compiled and ran, output all zeros.
- (round 003) `nisa.tensor_copy` on GpSimd cannot read PSUM (compile-time assertion).

- `nc_matmul(..., accumulate=(k > 0))` with several accumulation groups in one PSUM tile corrupted every group but
  the last (rel. error 0.25-1.0); the default `accumulate=None` (compiler-inferred) is correct. Never pass explicit
  accumulate flags.
- `/tmp/engine-gpu-48.lock` belongs to another user and fs.protected_regular=2 blocks opening it: run dev-eval with
  `GPU_LOCK_FILE=<session dir>/lab/gpu.lock` (still true in round 002).
- NKI parses kernel source: an `exec`-generated `@nki.jit` function fails (`entry function not found`), and tuple
  unpacking in a `for` target fails (`expecting simple variable`).
- A NEFF build's test run occupies its core: building on core 50 while profiling on 50 fails the capture
  (`cores busy`). Build on 51, profile on 50.
- `neuron-explorer` lives in `/opt/aws/neuron/bin` (not on the default PATH).
- Micro-benchmarks: >8k matmuls accumulating into one PSUM tile slowed a stream 2.5x (the documented cliff).
- The skill's `engine_client.py` does not pass `NEURON_RT_VISIBLE_CORES`; an engine started through it would claim
  every core. `lab/drive.py` (session dir) sets it.

## Round-001 next steps (superseded by round 002 above)

- Prefill 4096 attention (done: nkilib attention_cte). Decode NEFF fusion (done: 16 layers per NEFF).

## Round 005: what changed, measured

Tools (session dir `lab/`, not part of the engine): `one.sh NAME OUT` / `onev.sh NAME OUT "sed"` (one NEFF variant, from
a private copy of `kernels/` with a sed applied, test run on core 51), `prof.sh NEFF TAG [CORE]` (neuron-explorer
capture + json), `prof/dbins.py` (HBM->SBUF GB/s per time bin **per physical core**: DMA engines (e//16)%2; the
`dma` records carry `subgraph` sg00/sg01 = core), `prof/bins.py` (engine busy per core), `prof/ev.py` (instruction
list in a window; **only sg01's instructions are traced**, sg00 has ~1000), `runl.py`/`rt.sh FN B POS [sed]` (run a
decode kernel once on random inputs at a given position, save the output: bitwise check of a variant against
another), `A.sh`/`B.sh` (round-004 binary on core 50, candidate on 51). Profiles of two NEFFs captured at the same
time on cores 50 and 51 disturb each other (hbm_read_bytes and time both inflate): capture one at a time.

### The decode stream gap at attention (found with dbins.py)

Every decode layer had a ~25 us hole in the HBM weight stream after the QKV GEMV (both cores near 0 GB/s): the K/V
cache writes went through software DGE on GpSimd (~10 us), the cache loads on the Sync queue waited for them, and the
o_proj weight slots behind them in the Sync queue were not issued. Moving writes/loads to hardware DGE queues alone
did not close it: the compiler's scheduler issues a ring slot's DMA where it appears in program order, so the
o_proj slots were only issued after the attention code, and the ring (6 slots) runs dry during attention.

| change | measured | tokens |
|---|---|---|
| K cache [0, B) and V cache loaded at the start of the layer, before this token's K/V are written (stale at pos); this token's k patched into column pos of K^T with `tensor_copy` to a dynamic free-dim offset (`.ap(..., scalar_offset=pos (uint32 SBUF), indirect_dim=1)`); this token's v patched into row pos%128 of tile pos//128 of V: one-hot matmul builds e[p]*v[n] in PSUM, `scalar_tensor_tensor(x * (p != r) + that)` (exact: x*1+0, x*0+v), dynamic copy out of / back into the tile (partition-dim dynamic offsets are not supported). K/V writes to the HBM cache on the Scalar HWDGE queue, V as 4 contiguous 256-byte rows (PE transpose of the 4 v columns, exact) instead of 512 2-byte elements | (with the next row) grp16 b384 profile 11.389 -> 11.209 ms, b4224 12.036 -> 11.866 ms | bitwise (rt.sh at pos 5/300/2300/2559/4000; same.py 40/40) |
| o_proj ring slots prefetched in program order right after the QKV GEMV (`gemv_ring_prefetch`, `O_PRE = 6`; `gemv_ring(..., pre=)` then skips those DMAs) | see above | identical |
| embed fused into the first 16-layer NEFF (`dec_grp16e_B`), final norm + LM head + argmax into the second (`dec_grp16h_B`, the head streams through the layers' ring): 2 executions per step instead of 4 (`GRP_FUSE`, host `fuse_e`/`fuse_h`; `ENGINE_NO_FUSE=1` falls back to dec_embed + dec_layer x 32 + dec_head) | gh384 profile 12.840 ms vs 11.21 + 1.647 separate | same.py 40/40 |

**Pitfall:** a `scalar_offset` counts in units of the indexed dimension of the tensor's *shape* (offset 1 on dim 1 of a
[128, 8, 64] tile moves 64 elements), not in elements. Passing pos//128*512 for the tile of a [128, NTT, 512] V wrote
out of bounds and hung the core (NRT_TIMEOUT at bucket 2560). The register form (`preg`) is rejected for
`tensor_copy` dynamic dst (needs a uint32 SBUF tensor).

End to end (ab.py, 4 rounds, round-004 binary A on core 50 vs candidate B on core 51; B's core is 0.2-0.35% slower):

| candidate | decode 256 | 1024 | 4096 | prefill |
|---|---|---|---|---|
| K/V early + patch + o_proj prefetch | +1.29% | +2.00% | +0.86% | unchanged |
| + embed/head fused (this engine) | +1.91% | +2.76% | +1.60% | unchanged (+0.5 / +0.2 / 0.0%) |

**dev-eval 001 (2 rounds, clean build by dev-eval, engine with all of the above + HSK head split): PASSED, score
1.6015 (prefill 1.9012, decode 1.3490)**, 0 mismatching steps; prefill 30.6 / 107.1 / 449.5 ms, decode 1486.3 /
1478.4 / 1538.9 ms at 256 / 1024 / 4096 (round 004: 1518.6 / 1522.7 / 1570.3). The reference's decode times were
higher than in round 004's evaluation (1902-2136 vs 1873-2101 ms), which inflates the decode ratio; the engine's own
decode time improved by 2.1 / 2.9 / 2.0%.

Verification of the final source: same.py vs round 004 40/40 identical (fused engine and again with the head split);
`lab/longcheck.py` vs round 004: 200 + 3840, 4000 + 161, 4097 + 65, 4161 + 1, 3000 + 1 + 1099, 2049 + 20 + 2000,
1/2/129/513-token prompts, 255 + 10 + 10, 1023 + 5 + 60: every request returned all its tokens, all identical.

### Next ideas (not measured)

- The core-0 imbalance is worth ~3.5% decode (MLP_MOVE results) if a cross-core handoff that neither hangs nor
  costs a barrier can be found (e.g. an asymmetric gate/up / down split with a matching weight layout: not bitwise,
  the down K-split partial sums regroup; prefill shares the weight layout and would need the same split).
- Remaining decode stream dips (profiles): attention at large buckets (~10-20 us per layer at 4224, ring of 10 x 8 KB
  slots), exchange 2 waits.
- Prefill unchanged this round: S = 256 is PE-bound (TensorE 100% busy outside three ~20-30 us exchange gaps per
  layer); last layer could compute Q / attention for the last token only (~0.5-0.7% prefill).

### Physical core 0 is the slower core in every decode layer (measured end to end)

Profiles show core 0 (sg00, DMA engines (e//16)%2 == 0) streaming ~5% slower than core 1 and core 1 waiting
~50 us per layer at exchange 2. To rule out a profiler artifact: two diagnostic engines (buckets 384/4224 only) where
one core skips 6 of its 32 down-projection output tiles (11 MB of its 218 MB per layer, wrong tokens): decode 256
**1416 / 1420 ms when core 0 skips vs 1467 / 1464 ms when core 1 skips** (fused engine: 1488), the same on logical
cores 50 and 51. So the step time follows core 0's work; core 1 has slack. Same layer code and bytes on both cores;
the cause is not known (the head, a pure vocab-split GEMV, shows no measurable asymmetry: below).

| attempt | measured | status |
|---|---|---|
| `MLP_MOVE = j`: core 1 computes core 0's last j gate/up intermediate tiles (from core 0's weight run, same instructions) first and sends SiLU*up to core 0 with a gpsimd `sendrecv` (pipe 2) before core 0's down | j = 4: decode 256 **-3.6 / -3.8%** (1434 vs 1488 ms), j = 6 -3.4 / -3.6%, j = 2 -2.7%; bitwise (rt.sh, grp16e) | **hangs intermittently** (NRT_TIMEOUT in dec_grp16e/h): first with per-layer buffers in pid-specific code (sendrecv writes the peer's dst at the same SBUF address), then with one persistent buffer pair per NEFF; j = 6 ran 2/2, j = 4 hung 3/4. Disabled (`MLP_MOVE = 0`) |
| same on pipe 0 (shared with exchange 1 and the head's combine) instead of pipe 2 | - | hung 3/3 |
| same with `dma_engine=nisa.dma_engine.dma` (CoreBarrier-synchronized sendrecv) | never hung (3 runs), decode **+3.5% slower** than without the move | rejected |
| LM head vocab split 485 / 517 tiles instead of 501 / 501 (`HSK = 16`, host layout + idb; arithmetic per logit unchanged) | decode +0.06% / +0.37% (two core orders): noise-level | kept (bitwise, same.py 40/40) |

### Tried and rejected (round 005)

- Prefetching the next layer's first QKV slots before exchange 2 (`Q_PRE`) or the first gate/up slots before
  exchange 1 + norm (`G_PRE`): the compiler then re-reads weights (hbm_read 7.005 -> 7.66 / 8.88 GB per 16 layers)
  and the NEFF gets slower (11.21 -> 12.15 / 14.08 ms). Removed.
- 9 instead of 6 16-KB ring slots (b384) without the program-order prefetch: the gap only moved (0.745 vs 0.726 ms).
- K/V loads and writes on Sync / Scalar / GpSimd queues without the early load: 0.725-0.736 ms vs 0.733 (noise).

## Round 006: what changed, measured

Tools (session dir `lab/`, not part of the engine): `A.sh`/`B.sh` run `lab/bin/{A,B}` on logical cores 50/51; lab
builds use `NEFF_BUILD_CORES=51` (or 50). **Cores 48 and 49 belong to dev-eval** (the engine is started with visible
cores 48-49): a lab NEFF build test-running on 49 made dev-eval fail with `nrt_init ... cores busy`. A/A noise (round-005
binary on both cores, 3 rounds): prefill +-0.5%, decode B (core 51) 0.2% slower.

| change | measured | tokens |
|---|---|---|
| `MLP_SHIFT = 4`: core 0 computes gate/up for its first 52 intermediate tiles and the down K-split over them; core 1 computes its own 56, then core 0's last 4 (gate/up from core 0's `wgu` run, same instructions), then their down partial (`gemv_ring_rows` over core 0's `wd` run, K tiles 52..55, several output rows per strided DMA) into a second PSUM tile, added (fp32) into its own partial before exchange 2. Exchange gives P0 + (P1 + M) on both cores (two-operand fp32 add commutes: same bytes). No extra sync point (round 005's MLP_MOVE needed a separate sendrecv and hung). Weight layout and prefill unchanged. | ab 4 rounds vs round 005 (B on core 51): decode +3.44 / +2.63 / +2.25% at 256 / 1024 / 4096, prefill unchanged | **not bitwise** vs round 005 (down-proj fp32 grouping): 3 of 12 ab streams on nonce prompts diverge (first at token 6-18); dev-eval 0 mismatching steps |
| shift 2 / 3 / 5 / 6 vs 4 (ab 3 rounds, 256 / 4096) | 2: -0.1 / -1.8%; 3: +0.5 / -0.5%; 5: -0.5 / -0.8%; 6: -1.7 / -2.6% | |

dev-eval 002 (1 round): PASSED, score 1.6649 (prefill 1.919, decode 1.444), 0 mismatching steps; engine decode 1433.0 /
1437.4 / 1501.4 ms (round 005 evaluation: 1486 / 1478 / 1539); the reference's decode times were again higher than in
the round-005 evaluation (1996-2195 ms), inflating the ratio. same.py candidate vs a second build of it: 40/40
identical, shapes self-consistent, no hangs.

**Final source, dev-eval 003 (3 rounds, clean decode rebuild by dev-eval): PASSED, score 1.6208 (prefill 1.9004,
decode 1.3824)**, 0 mismatching steps in every round; prefill 30.6 / 107.1 / 449.7 ms, decode 1433.2 / 1437.3 /
1500.9 ms at 256 / 1024 / 4096 (round 005: 1486 / 1478 / 1539); reference decode 1883-2192 ms.

### Tried and rejected (round 006)

- Prefill attention: exp of all of a query group's key tiles in one `activation_reduce` (one row-sum accumulator,
  not bitwise) instead of one per 512-key tile: prefill +0.6 / +0.6 / -0.1% (256 / 1024 / 4096, ab 4 rounds):
  noise; the 4096 attention is not bound by per-instruction ACT overhead. Not kept.

### Next ideas (not measured)

- Prefill 256 (`pf_layer_ns`) is weight-stream bound too: the same core-0 imbalance may apply (shift o_proj/down
  N-split tiles to core 1).
- The optimal shift may depend on the bucket (3 looked best at 256, 4 at 4096): per-bucket `MLP_SHIFT`, < 0.5%.

## Round 008: what changed, measured

Tools (session dir `lab/`, not part of the engine): `one.sh NAME OUT [KDIR]` / `onev.sh NAME OUT "SED" [FILE]` (one
NEFF variant from `lab/k/`, sed applied to FILE, default prefill.py; `CORE=50` for the test run), `prof.sh NEFF TAG
[CORE]` (neuron-explorer capture, 3 executions, summary of the 3rd; `view --output-format json` then writes
`ntff.json`), `tools/act.py` (engine busy % per time bin per physical core, from `active_time`), `tools/waits.py` /
`tools/win.py` (sg01 instruction list and semaphore waits in a window), `tools/dbins.py` (HBM->SBUF GB/s per bin per
core from the `dma` records), `tools/phase.py`. `A.sh`/`B.sh`/`C.sh` run `lab/bin/{A,B,C}` on logical cores 50/51/50.
**Lock:** `/tmp/engine-gpu-48.lock` (the session's GPU_LOCK_FILE) belongs to another user: export
`GPU_LOCK_FILE=<session>/lab/gpu.lock` for ab.py / same.py / dev-eval (as in rounds 002-006).
**Builds:** editing `build_neffs.py` changes every NEFF's cache key (full ~15 min rebuild); prefill-only edits rebuild
the 22 prefill NEFFs (~8 min with 6 jobs).

### S <= 256 prefill layer (`pf_layer_ns`): exchange gaps (profile 0.892 -> ~0.869 ms per layer)

Profile of the round-006 layer (0.890 ms): PE idle ~100 us per layer: start (x load + norm, ~25 us), attention-output
exchange (~15-18 us), h1 exchange (~25 us), MLP-intermediate exchange (~31 us: last store ~11 us behind the down-weight
prefetch, barrier ~3 us, 3.6 MB load ~17 us), end (~5 us).

| change | profile per layer | e2e |
|---|---|---|
| `NS_SHIFT` (gate/up tiles moved from core 0 to core 1; ms exchange covers any split): 1 / 2 / 3 | 0.8856-0.887 / 0.888 / 0.898-0.91 vs 0.890 | shift 1: -0.5% (noise, B on core 51): not kept (`NS_SHIFT = 0`) |
| `NS_OWNF`: own heads' attention output kept in SBUF (`acb`, stage_wb writes it there) and only the other core's 16 heads loaded after the barrier; o_proj and down accumulate the own K half first, 4 output tiles at a time (`NS_NB`), so the other half's load overlaps the own-half matmuls. Core 0's order is unchanged (bitwise), core 1's fp32 accumulation order changes (as round 006's MLP_SHIFT) | 0.884 | |
| attention stores: one DMA per finished kv group (4 heads, from `acb`) on the GpSimd queue instead of 32 per-unit 256-byte-row DMAs on the Scalar queue: the Scalar engine was stalling ~19 us per store on DMA-queue credits, delaying the exps | 0.877 | **p256 prefill +2.9%** (ab 8 rounds, 30.7 -> 29.8 ms) |
| `NS_SQ_EARLY`: own half of h1 squared before the h1 exchange (elementwise, same bytes) | ~-0.2% | |
| `NS_DVSQ`: half of the norm's squares as x*x on the Vector engine (x^2 exact in fp32, RNE to bf16 on both engines) | 0.8736-0.8752 vs 0.8760-0.8771 | |
| `NS_PSX`: the MLP norm from exchanged partial sums of squares (each core: own 16 hidden tiles -> fp32 row -> `xa` scratch, barrier, the other's row broadcast-loaded; rstd from P0+P1, the same on both cores), own half of xn2 and gate/up on the own K half (`NS_NBG = 2` tiles per batch) before the h1 exchange completes. Not bitwise (norm sum grouping; gate/up order on core 1). Needs `xa` bound to pf_layer_ns (host: bind any of xa/xb/hs/ms the NEFF takes) | 0.869 | **p256 prefill +3.6%** (ab 10 rounds, 30.9 -> 29.8 ms) |

| `NS_POSTR` (S <= 256, both norms of the layer): the GEMMs after a norm take bf16(x * g) (`_scale_g`, alternating Vector `tensor_scalar` / Scalar `activation(copy, scale=g)`), and the per-token rstd multiplies the fp32 GEMM result at the PSUM evacuation (`tensor_tensor(ps, rstd)` instead of `tensor_copy`), so the squares -> sum -> rstd chain and the 32 scaling STTs (15 us) are off the critical path; in the MLP norm the partial-sum and h1 exchanges are emitted after the first gate/up batch's own-half matmuls (`_ns_psx_exch`). **Not bitwise and a different rounding point than NxD** (bf16(rstd * W.bf16(x*g)) vs bf16(W.bf16(x*rstd*g))): 1-ulp differences in a fraction of q/k/v/gate/up values | ~0.863 | **p256 +0.8%** on top (ab 10 rounds vs the previous candidate, 29.7 -> 29.5 ms); dev-eval 002: 0 mismatching steps |

Rejected (profile per layer): exchange loads on the GpSimd / Scalar queues instead of Sync (0.889-0.892 vs 0.884;
later 0.872-0.884, noise); norm scaling partly on GpSimd (tensor_scalar + tensor_tensor, 2 ops: 0.907-0.930); norm
scaling by token halves with the first GEMM tile in two 128-column passes (64 narrow STTs: 0.882-0.885 vs 0.877-0.880).
`scalar_tensor_tensor` has no `engine` argument. QKV tiles accumulated kt-major right after the norm (`NS_KTM`, 4 or
6 PSUM tiles, bitwise): 0.870-0.875 vs 0.872-0.876 (noise; the first ~15 us of every layer NEFF is the execution
preamble: engine sync, semaphores). Exchange stores (h1 / ms / partial sums) on the GpSimd or Sync queues instead of
Scalar: noise. With NS_POSTR, the h1 / ms exchange loads on GpSimd: 0.864-0.880 vs 0.862-0.865 (the compiler still
issues the barrier-dependent h1 load on Sync ahead of the first gate/up half-slabs, so the TensorE waits ~10 us there;
not solved). Two layers per NEFF (`pf_pair_ns`: second layer with its own exchange scratch, core barrier on the
intermediate x; host: 15 pairs, layer 30, last layer): profile 1.731 ms vs 2 x 0.871 per pair, **e2e p256 0.999x**
(queued executions overlap their preambles: per-execution cost is not on the critical path). Removed.

### Prefill attention: row max chained in the Vector accumulator (`CHAIN_MAX` in pattn.py, all buckets)

stage_qk's mask instructions (range_select / tensor_scalar_reduce, one per 512-key tile) each wrote their tile's max
(`reduce_res`, a DVE_READ_ACCUMULATOR of ~84 ns per tile) and a tensor_reduce took the max over those. Now the first tile
resets the Vector engine's accumulator (`reduce_cmd.reset_reduce`), the others accumulate (`reduce_cmd.reduce`,
`reduce_res=None`) and only the last reads it out; max is exact in any order, so the bytes are the same (same.py vs the
previous engine: 40/40 identical). The 4096 attention is DVE-bound (DVE time in the attention window: masks 65%, read-
outs 8%, tensor_tensor 9%). Profile per layer: 4096 14.37 -> 14.30 ms, 1024 3.311 -> 3.290 ms, 256 unchanged; **ab 6
rounds: prefill +1.0% at 1024, +0.5% at 4096**, 256 +0.1%.

Rejected: every 2nd / 3rd / 4th key tile's P^T on the TensorE instead of DMA transposes at 4096 (the DMA transposes
run at ~250 GB/s per core): 14.65 / 14.49 / 14.42 vs 14.35 ms (the extra PSUM->SBUF copies load the DVE, which is the
attention's bottleneck).

### Last prefill layer, S >= 1024 (`LAST_Q`, `_last_qattn`)

Only the last prompt token's output is used: K/V for all tokens as before, but Q and attention only for the 128
queries ending at pos (window start pos-127 at a dynamic offset: register from an int32 subtract; .ap offsets must be
non-negative), all key tiles with `range_select` against a per-row bound pos-127+p, exp over full 512-key tiles
(extra keys masked to 0), only column pos stored to `att`. same.py vs the previous candidate: buckets 2048 and 4096
identical; bucket 1024 was wrong at first (`S // LT` = 0 PV tiles: fixed to a ceiling with a tile guard).
pf_last_4096 profile 4.46 ms before, pf_last_1024 1.40 ms before.

### Tried and rejected for prefill 4096 (attention overlap; per-layer profile, baseline 14.35 ms)

The S = 4096 layer: phase A (QKV, PE-bound) 1.5 ms, attention 2.25 ms (ACT 82%, DVE 78% busy: each 512-key tile
costs ~0.68 us on DVE (mask copy / max from PSUM) and ~0.71 + 0.28 us on ACT (exp + accumulator read-out)), then
o_proj/MLP 10.5 ms at PE 100%.

| variant | per layer |
|---|---|
| phases A+B by heads (core c: QKV of its 16 q / 4 kv heads for all tokens, no K/V/Q barriers), attention units of each finished (chunk, kv group) interleaved into the following QKV GEMMs (DMA transposes) | 14.67 |
| same, PE transposes / distances (1,2,8,9) / stores off the Scalar queue | 14.60-14.81 |
| by kv group: QKV of group g for all tokens (x normalized once into a private-HBM scratch) beside the attention of group g-1 (no shared data between the two streams) | 14.93; PE transposes 14.98; other distances 14.91-15.00 |
| non-diagonal tiles: max-only `tensor_reduce` from PSUM, QK recomputed and exp straight from PSUM (bitwise) | 14.65: a reduce-only pass from PSUM costs the same 0.685 us as copy+reduce (DVE and ACT are PSUM-read bound) |

Why the overlap fails: the DMA transposes (one SB->SB queue) slow down when they share the DMA engines with the
weight stream and q/k/v traffic (PV waits of 26-50 us), and the power governor throttles more when PE, ACT and DVE are
all busy (throttled time 0.95 ms -> 1.53 ms per layer at a 50% utilization limit). With PE/ACT/DVE each holding
~2.2-2.4 ms of work in A+B, the remaining bitwise lever is small.

### Decode (measured, not changed)

dec_grp16e_384 profile: each physical core streams ~337-345 GB/s in steady bins (~680 GB/s for the logical core, one
HBM stack); about twice per layer core 0 (sg00) dips to 90-260 GB/s for ~10 us while core 1 rises to 370-420, at
points where core 1 is in a norm / exchange. Swapping the cores' roles (pid -> 1 - pid) made the NEFF slower (10.85 ->
11.54 ms) with the stalls moving to the core that now had less work: the slowness belongs to physical core 0, not to
its role. Core 0's instructions are not traced, so the dips could not be attributed further. e2e decode is ~670 GB/s;
the remaining headroom is ~1.5%.

### Result (round 008)

dev-eval 001 (2 rounds, without NS_POSTR): PASSED, 0 mismatching steps, score 1.6285 (prefill 1.9273, decode 1.3761);
prefill 29.5 / 106.8 / 446.5 ms (round 006: 30.6 / 107.0 / 449.6).
**dev-eval 002 (2 rounds, this engine): PASSED, 0 mismatching steps, score 1.6317 (prefill 1.9369, decode 1.3746)**;
prefill speedups 2.356 / 2.270 / 1.358 (round 006: 2.266 / 2.262 / 1.351), engine prefill 29.3 / 106.65 / 446.6 ms,
engine decode unchanged (1434.1 / 1437.4 / 1501.3 ms; the decode ratio moved with the reference's times only).
same.py: the LAST_Q change is identical to its parent on all 40 cases (lengths 1..4097, 3 shapes each). This engine vs
the dev-eval-001 engine: 33/40 identical, every engine self-consistent; the 7 that differ are all prompts of 2..128
tokens (5 nonce, 2 random-token prompts: near-ties), first difference at token 1..21 (NS_POSTR's rounding point).

**Final engine (+ CHAIN_MAX), dev-eval 004 (3 rounds, clean build by dev-eval): PASSED, 0 mismatching steps in every
round, score 1.6357 (prefill 1.9425, decode 1.3774)**; prefill speedups 2.345 / 2.288 / 1.366 (round 006: 2.266 /
2.262 / 1.351); engine prefill 29.3 / 105.8 / 444.7 ms (round 006: 30.6 / 107.0 / 449.6, i.e. -4.2 / -1.1 / -1.1%),
engine decode 1433.9 / 1437.7 / 1500.8 ms (unchanged). dev-eval 003 (3 rounds, before CHAIN_MAX) had read p256 2.331
with a 1% faster reference: the speedups move by ~1% with the reference's own times.

### Next ideas (not measured)

- S <= 256: the h1 exchange still idles the TensorE ~10-15 us per layer (the barrier-dependent h1 load issued ahead of
  the gate/up weights on Sync), the MLP-intermediate exchange ~10 us (last store latency behind the down-weight
  prefetch + barrier), the attention-output exchange ~10 us. A way to make the compiler issue the weight slabs first
  (or a store/load path that does not share the Sync queue) would buy ~2-3% of p256.
- S = 4096 attention: per 512-key tile ~0.68 us DVE (PSUM read) and ~0.99 us ACT; transposes run at ~250 GB/s per core
  through one SB->SB queue. Non-bitwise chained row sums would cut ACT ~25% but round 006 measured no e2e gain alone.
- Decode: core-0 stream dips (~1.5% of decode) at norm / exchange points; not attributed (core 0 not traced).

## Round 009: what changed, measured

Tools (session dir `lab/`, not part of the engine): `attn_lab.py` + `run.sh VARIANT S TAG [CORE]` (one attention kernel
alone on random q/k/v, sink-like first key, saved output; `old` = attention_heads, `new` = attention_t), `check.py S
outs..` (rel. error vs an fp32 numpy softmax attention), `cmp.sh TAG` (max diff vs the old kernel's output: **always
check it; a pipeline race shows up as a few garbage heads**), `var.sh TAG "SED"` (variant of pattn.py: build, run on
core 51, check, profile on 50), `rep.sh TAG..` (profile twice more: single captures are sometimes 1.2-2x slow on this
shared machine), `one.sh NAME OUT [KDIR]` (one engine NEFF), `prof.sh`, `ops.py` / `lines.py` (opcode and per-source-line
instruction time; `nki_source_location` is in the instruction records), `stalls.py` (TensorE gaps by the instruction
pair around them), `busy.py`, `tl.py` (engine busy % per time bin per physical core from `active_time`),
`longcheck.py`, `scores.py` (CPU, transformers: attention-logit ranges of the model). Lock: `GPU_LOCK_FILE=lab/gpu.lock`.
Baseline binary for ab/same: the round-008 harness build (`runs/.../build/target/engine`); its `neffcache` can be
copied into `engine/target/neffcache` (same keys) to skip rebuilding unchanged NEFFs.

### Prefill attention for S >= 2048: `attention_t` (pattn.py)

Old attention (`attention_heads`, bitwise nkilib) at 4096: 2.16 ms alone, 2.25 ms in the layer; ACT 97%, DVE 95%
busy (per 512-key tile: DVE mask/max from PSUM, ACT exp + accumulator read-out, P^T by DMA transposes), TensorE idle.
New design: per (head, 512-query chunk j, 128-key block kb <= 4j+3) unit:
- S^T [128 keys, 512 q] = matmul(stationary K^T block, moving Q^T chunk) into one half of a two-bank PSUM tile;
- P^T = exp(S^T) straight from PSUM to bf16 SBUF, one ACT instruction per unit pair (1024 elements: the ACT costs
  ~170 ns + 0.84 ns/element per instruction); **no row max** (see below); diagonal blocks masked after the exp by GpSimd
  `affine_select` (P^T[key p, query c] = 0 for p > c);
- O^T [128 d, 512 q] += matmul(stationary V block, moving P^T) (P^T is already the moving operand: no transposes);
- row sums: bf16 P^T of 2 consecutive key blocks (4 for non-diagonal quads) added on the Vector engine, then one
  matmul(stationary ones [128,128], moving that sum) accumulating into a PSUM row-sum tile (broadcast on all partitions);
- chunk end: both PSUM accumulators copied to SBUF by the Vector engine first (frees the banks for the next chunk),
  reciprocal (3.4 us per [128,512], from SBUF), multiply, bf16 store to att.
Pipeline: S^T of unit u, exp of the pair ending at u-2 (`TB`), PV/sums of unit u-3 (`TD`).

**Why no max is safe here** (measured, `lab/scores.py`, HF transformers on CPU, bf16): scaled attention logits of all
layers for 4096-token prompts (3 random-token seeds, one English text): row max over keys in [-14.3, +34.8] for random
tokens, [-1.2, +67.1] for text, but **+67 only in layer 31**; layers 0-30 stay in [-14.3, +24.4]. fp32 and bf16 exp
overflow at 88.7, P sums stay far below fp32 range. Layer 31 at S >= 1024 is `pf_last_tok` (LAST_Q path, old
attention with the max), so attention_t never sees it. If a future change routes the last layer through attention_t,
add a shift (e.g. a per-head constant bias in the exp).

Degenerate prompts (lab/scores2.py, 2048 tokens, layers 0-30): one token repeated: row max in [-18.0, +14.0]; BOS
repeated [-1.2, +4.5]; a 7-token cycle [-16.4, +11.3]; token ids 0, 1, 2, ... [-11.7, +18.1]. Margin to 88.7 stays large.

Numerics vs NxD: scores not rounded to bf16 before the exp, no max shift, row sums of bf16 P (with bf16 pair/quad
partial sums) instead of fp32 sums of the exps. Alone: rel. L2 vs fp32 attention 0.00167 (old kernel 0.00158), max
|diff| vs the old kernel 0.0156 (1 bf16 ulp at |x|~2-4) at 2048 and 4096.

| step (attention alone, S = 4096, profile) | ms |
|---|---|
| old attention_heads | 2.16 |
| S^T layout, ones-matmul sums, mask by an identity-stationary matmul of a -30000 tile, reciprocal + multiply | 2.10 (TensorE 94%: every matmul pays a ~53 ns LDWEIGHTS, not hidden) |
| pipeline distances exp 2 / PV 4 | 1.95 |
| pairs of P^T tiles added in bf16 on DVE, one sums matmul per pair | 1.76 |
| one exp per unit pair over a two-bank PSUM tile | 1.72 |
| fresh PSUM accumulators per chunk (see the race below), 4-tile bf16 sums for non-diagonal quads | 1.64 |
| diagonal mask by GpSimd affine_select after the exp instead of the mask matmul | 1.63 (neutral, fewer PE ops: kept) |
| chunk epilogue copies PSUM accumulators out first (DVE), reciprocal from SBUF | 1.52 |
| pipeline distances exp 2 / PV 3 | 1.435 (in the layer: same as distance 4) |

Rejected / pitfalls (attention_t):
- Row sums on the Vector engine as fp32 elementwise accumulation: 3.0 ms of DVE (0.36 elem/ns for fp32 + bf16 -> fp32).
- 1/sum by `tensor_tensor(op=nl.power)` with -1 on GpSimd: 85 us per [128,512]. `nl.divide` is not a supported ALU
  op (tensor_tensor / tensor_scalar both reject it). ACT `activation(op=nl.reciprocal)`: wrong values (max diff 3.1)
  and 476 ACT_TABLE_LOADs. ACT copy of the O^T accumulator in the epilogue: 2.08 ms vs 1.52 (table switch).
- Bf16 adds on GpSimd instead of DVE (quad level or half of the pair adds): 1.55-1.78 vs 1.52 ms.
- **Persistent PSUM accumulators (two pairs alternating between chunks) with explicit `accumulate=` flags: fast
  (1.47 ms) but wrong**: correct at one pipeline depth, garbage heads (|x| ~1e17) at others. Use fresh tiles per chunk
  (compiler-inferred flags). Same family as the round-001 pitfall.
- **An exp over a PSUM range that no matmul of the pair wrote (the unused head of a diagonal pair's second half)
  produced garbage at some pipeline depths**: exp only the written ranges.
- Deeper pipelines (exp distance 3, PV distance 5-8) starve PSUM (8 banks: 2 per pair tile + 2 accumulators + the next
  chunk's 2) and serialize: 1.9-3.2 ms.

In the layer (pf_layer_4096 profile): 14.30 -> 13.55 ms. Attention now ~1.45 ms of the layer at TensorE 80-86% busy.
Bucket 1024 with pf_layer_tok + attention_t: 3.285 ms vs pf_layer_pipe 3.295: no gain, the 1024 bucket keeps the pipe.

### MLP norm squares streamed per o_proj output tile (bitwise)

Phase C squared the whole h1 chunk (one ACT op over [128, 32, 256]) and summed it (32 ones-matmuls) after the last o_proj
tile, so gate/up waited ~20 us per chunk. Now each h1 tile is squared and its ones-matmul accumulated as soon as
o_proj produces it (same instructions per element, same accumulation order: same.py 40/40 identical vs the
attention-only build). pf_layer_4096 13.565 -> 13.457 ms, pf_layer_1024 (pipe) 3.289 -> 3.264 ms.
Rejected: next chunk's x load + input norm emitted 8 or 24 QKV tiles into the current chunk of phase A: 13.44-13.47 vs
13.45-13.47 ms (neutral; the phase-A TensorE dips are not the norm).

### Result (round 009)

ab.py (4 rounds, round-008 binary on core 50, candidate on 51): prefill 29.7 -> 29.5 / 106.1 -> 105.3 / 445.2 ->
420.3 ms (+0.2 / +0.8 / **+5.9%**), decode unchanged (1434 / 1438 / 1502 ms). same.py vs the attention-only build:
40/40 identical. `lab/longcheck.py` (200 + 3840, 4000 + 161, 4097 + 65, 4161 + 1, 3000 + 1 + 1099, 2049 + 20 + 2000,
prompts 1/2/129/513/1025/2047/2048/4095/4096): every request returned all its tokens; vs round 008, 13/15 identical,
3000+1+1099 differs from token 544, 4095+5 from token 1 (attention numerics).
**dev-eval 002 (2 rounds): PASSED, 0 mismatching steps, score 1.6577 (prefill 1.9852, decode 1.3841)**; prefill speedups
2.346 / 2.305 / **1.447** (round 008: 2.340 / 2.288 / 1.364); engine prefill 29.35 / 105.2 / 420.0 ms.

**Final source, dev-eval 003 (3 rounds, built by dev-eval): PASSED, 0 mismatching steps in every round, score 1.6545
(prefill 1.9796, decode 1.3828)**; prefill speedups 2.328 / 2.305 / **1.446** (round 008: 2.340 / 2.288 / 1.364);
engine prefill 29.5 / 105.1 / 420.1 ms, decode 1434.0 / 1438.2 / 1501.4 ms (unchanged).

Clean-checkout build (`lab/clean`, files of `git ls-files`, empty target/): 61 NEFFs in ~25 min, `ldd` shows libnrt +
libc/libstdc++/libm/libgcc_s only. Final source vs the dev-eval-002 build: same.py 40/40 identical.

### Where the time is now (S = 4096 layer, 13.46 ms, profile)

Phase A (QKV, 1.5 ms, TensorE 89-100% with dips at chunk starts), attention (~1.45 ms, TensorE 77-86%: PSUM-limited
pipeline between S^T, exp and PV), phase C (10.5 ms, TensorE ~98-100%). GEMM floors: phase A 1.32 ms, phase C 10.1 ms,
attention_t TensorE work ~1.2 ms.

More attention_t variants measured and rejected (alone, S = 4096, vs 1.43 ms): 256-query chunks (a unit pair in one
PSUM bank, so deeper pipelines fit): 1.95 / 1.88 / 1.68 ms at distances 2/3, 4/6, 6/9 (more, smaller instructions);
Q of the next head and K/V of the next kv group loaded one chunk ahead: 1.429 (neutral, not kept). Row sums with
queries on partitions (stationary = the P^T block, moving = a ones column, accumulator [128, 4]; 1/sum on [128, 4],
broadcast back by stride-0 nc_transposes as stage_wb does): correct but 1.615 ms (one-column matmuls cannot hide
their LDWEIGHTS: 4 x 53 ns per sums event).
The profile shows LDWEIGHTS overlapping the previous matmul when that one streams >= 256 columns (512-column S^T
matmuls issue every 215 ns = 512 x 0.42 ns), so the TensorE is not the bound: all three engines sit at ~70% busy,
limited by the S^T -> exp -> PV chain and the 8 PSUM banks.

Other measurements this round: decode request time is linear in the token count (22.40 ms x n + 0.35 ms for
n = 1 / 8 / 32 / 64): no per-request overhead to remove. Decode NEFF profiles: dec_grp16e_384 10.84 ms for 7.01 GB,
dec_grp16h_384 12.41 ms for 8.06 GB (~650 GB/s; the profiler's hbm_ddr_bandwidth figure is 716 GB/s). Phase A of the
4096 layer has only ~71 us of TensorE gaps > 300 ns (sg01): it is near its GEMM floor. In the 1024 pipe layer the
first own chunk's attention runs alone (pattn window ~393-659 us of 3.26 ms) but o_proj starts under it; the TensorE
idles only ~60 us there.

### Next ideas (not measured)

- attention_t: the S^T -> exp -> PV chain is PSUM-limited (8 banks); anything that frees banks earlier helped
  (the epilogue copy-out: 1.64 -> 1.52 ms). The DVE reciprocal (3.4 us per [128, 512], 128 per core) is the largest
  single DVE item left; no fast exact alternative found (divide / power / ACT reciprocal / transposed sums, above).
- 1024 bucket: the first own chunk's attention in pf_layer_pipe runs through the old stages; a chunk version of
  attention_t (all 32 heads, 256 queries, output into the o_proj SBUF tile) would cut its ~60 us of TensorE idle.
- Overlap attention_t (ACT / DVE heavy) with phase A by heads (no K/V/Q barriers); round 004's version of this failed
  with the old attention (DMA transposes); the new one has none.

## Round 010 (rejected candidate, notes carried over by round 011): measured where the remaining time goes

Tools (session dir `lab/`, not part of the engine): `drive.py ENGINE_WRAPPER LENS REPS` (split requests, prints
prefill / decode ms and a token hash), wrappers that set env switches (`E.sh` sources `e.env`), `one.sh KDIR NAME TAG`
(one NEFF from a kernel copy, test run on core 51, random-data capture on 50), `peana.py` / `pecore.py` (TensorE
matmuls per core: `fully_qualified_subgraph` /sg00 vs /sg01 separates the physical cores in the `instruction` list;
ideal time = moving columns / 2.4 GHz, the profile's `pe_clock_freq`), `where.py` / `win.py` (gaps, HAM windows,
instruction list in a time window), `hamloss.py`, `attn/` (attention_t alone, bitwise check vs the round-009 kernel).

### New host diagnostics (env-gated, default behaviour and tokens unchanged; same.py 40/40 vs round 009)

- `ENGINE_STEPTIME=1`: wait for and log every execution (`exec NAME ms`). With it,
  `ENGINE_PROF_NEFF=name ENGINE_PROF_IDX=k ENGINE_PROF_OUT=file.ntff` profiles the k-th execution of that NEFF
  **with the real weights and activations** (`nrt_profile_start/stop`); view it with
  `neuron-explorer view -n <cached neff> -s file.ntff`. Random-data captures (`neuron-explorer capture`) of the 1024
  layer were throttled to 0.42-0.73 util limit and read 4.1-7.4 ms; the real-data profile reads 3.26 ms.
- `ENGINE_SKIP=prefix,prefix`: skip NEFFs by name (timing only, wrong tokens); `ENGINE_PF_PATTERN=ELPH..` runs an
  execution pattern (E embed, L layer, P pick, H head) instead of a prefill; `ENGINE_DEC_PATTERN=ee|hh` runs a decode
  step as two grp16e / two grp16h executions; `ENGINE_VERBOSE=1` logs parse / generate / total ms per request.
- `host/json.h`: allocation-free fast path for integers: request parse 0.27 -> 0.14 ms at 4096 tokens (0.03 ms at
  256). That is the only change in this round's engine.

### Where the time is (measured, real data unless noted)

| item | measured |
|---|---|
| p256 request 29.3 ms | 31 layers 26.58 ms (0.857 each), last layer +0.86, dec_head +1.68, embed +0.14, host ~0.05 |
| first execution of a request | ~0.2 ms launch + completion latency of `nrta_execute_schedule` (one 37 us embed alone: 0.25 ms request); synchronous `nrt_execute` is slower (0.43 ms) |
| NEFF switch in a queue | ~40-45 us each time the model changes between consecutive executions (EPEPEPEPEP 0.78 ms vs EEEEEPPPPP 0.49; LELELE.. vs LLLLLLEEEEEE +43 us per switch; LHLHLH vs LLLHHH +45 us); same NEFF back to back: none. Decode: (ee + hh)/2 vs eh = 37 us per step, so a single 32-layer decode NEFF would buy ~0.2 % (not built) |
| pf_layer_1024 | 3.26 ms; TensorE active 98.9 %; ideal (moving columns at 2.4 GHz) 2.96 / 2.91 ms per core; 256-column matmuls issue every 105-110 ns (= peak); lost: preamble ~21 us, gaps > 300 ns 140-190 us (attention stages, phase boundaries), HAM half-rate windows |
| pf_layer_4096 | 13.44 ms; ideal 12.49 per core (attention S^T 450 + PV 450 + sums 137 us of it); gaps > 300 ns ~510 us (300 of them in attention_t: S^T waiting for PSUM / exp) |
| pf_layer_256 | 0.866 ms; TensorE 94.5 %; ideal ~0.74; gaps > 1.5 us ~40 us per core (exchange waits: o_proj other half 6-9 us, gate/up ~12-15 us, down 7-10 us x2) |
| decode grp16e_384 | 10.80 ms profiled, 7.005 GB, DMA active 99.7 % (e2e 667 GB/s of the 716 GB/s `hbm_ddr_bandwidth`) |

**HAM (TensorE half-rate windows):** the `ham` records of a profile mark 3.41 us-quantized windows where the PE runs at
4/8 rate (`type 1`, k=4/n=8; matmul issue interval 212 ns instead of 107). They follow PE idle periods longer than ~3 us
and low-activity phases (attention): 65-73 us per core per p256 layer, 133-137 us per 1024 layer, 283-300 us per 4096
layer (= the `throttle_activity_1` 0.5 limit and `throttle_avg_util_limit` 0.93 / 0.97 / 0.986). Matmul time lost inside
them (upper bound): 17-21 / 35-40 / 80-84 us per layer (2.2 / 1.1 / 0.6 %). A micro-benchmark (`lab/ham`) confirms:
a ~5 us PE idle gap is followed by 6.8 us at half rate; filling the gap with independent dummy matmuls (zeros work as
well as data) removes most windows (226.6 -> 222.0 us), but only if the PE stays busy through the whole wait (5 dummies
at the start of the gap: no effect).

### Tried and rejected (numbers; pf_layer_256 profile baseline 0.861-0.866 ms unless noted)

- Dummy matmuls in the exchange waits: free-standing (`_warm`, 600 per layer): 0.894 ms (they ran before the waits and
  delayed the real work; throttle limit 0.922 -> 0.947); zero matmuls accumulated into the waiting PSUM group (pinned
  in the accumulation chain, `+0.0` exact): 0.875 (the waits are longer than where the pin puts them). Not kept.
- NS_NBG 3 / 4 (more own-K-half gate/up work before the h1 exchange): 0.863-0.868 (the PE still stops after ~42 matmuls:
  it waits for the other half's scaled xn2, which needs the barrier + 1 MB h1 load (~5.4 us, 190 GB/s next to the
  weight stream) + `_scale_g`).
- h1 exchange load on Scalar (NS_XQH=2) 0.867-0.869, on GpSimd software DGE 0.868; MLP-intermediate load on Scalar
  0.869-0.871; h1 load split over Sync + Scalar 0.865-0.870. HWDGE DMAs accept only the Sync / Scalar engines.
- NKI backend optimizations (`_enable_backend_opt`, i.e. neuronx-cc instruction scheduling + linear-scan allocation;
  the standalone NKI route disables them by default): 0.858-0.867 (noise).
- attention_t epilogue (alone, S = 4096, 1.428-1.431 ms): 1/sum on the transposed [128 q, 4] sums (exact PE transposes
  of row 0 and back, bitwise identical output): 1.62 ms inline, 1.53-1.56 spread over later units (the compiler put the
  broadcast `tensor_scalar` on the Scalar engine: +249 us ACT; forced to Vector: 1.53); ED 6 / 12 units: 1.473 / 1.477.
  The PE-side transposes wait on DVE results and stall the in-order S^T stream. 1/sum as rsqrt^2 on GpSimd: 1.817 (not
  bitwise, GpSimd shares the SBUF port with DVE). PSUM matmul output in bf16 is rejected on gen3.
- Runtime env vars (decode and prefill unchanged within noise): NEURON_RT_DISABLE_EXECUTION_BARRIER,
  DBG_EARLY_COMPLETION, DBG_EARLY_POSTING, DBG_ENABLE_BRANCH_PREFETCH, ONE_THREAD_PER_CORE,
  LOW_LATENCY_TASKS_CPU_AFFINITY, DBG_DMA_PACKETIZATION_SIZE (2048 / 16384 / 65536), DISABLE_PACK_DESCRIPTORS,
  DBG_DMA_QOS_SERVICE_LEVEL.
- No cold-start effect: the first 256 / 1024 / 4096 requests after one warm-up equal the later ones.

### Remaining levers (estimated, not built)

- Prefill tail: dec_head fused into pf_last_S (one switch, head through a ring: ~0.1 ms), embed fused into layer 0
  (~0.1 ms), Q / attention for the last query block only at S <= 256 (~0.1 ms): about +1 % at p256, +0.3 % at 1024.
- Decode: one 32-layer NEFF per bucket (~0.2 %); attention at bucket 4224 overlapped with the weight stream (~1 % of
  p4096 decode).
- Every case is within ~5-8 % of its hardware floor (TensorE at 2.4 GHz, HBM at 716 GB/s); none of the above reaches the
  acceptance thresholds alone.

### Result (round 010)

dev-eval 001 (2 rounds, built by dev-eval): PASSED, 0 mismatching steps, score 1.6440 (prefill 1.9818, decode 1.3638);
engine prefill 29.4 / 105.0 / 420.0 ms, decode 1433.7 / 1437.9 / 1503 ms: the round-009 engine's numbers (the kernels
are unchanged; same.py vs the round-009 binary 40/40 identical).

## Round 011: what changed, measured

Tools (session dir `lab/`, not part of the engine): `drive.py WRAPPER LENS REPS [OFF] [DECODE]` (split requests, ms +
token hash), `one.sh NAME OUT [KDIR]` (one NEFF from a kernel copy, test run on core 51), `relink.sh OUT NAME=FILE..`
(link the engine with some NEFFs replaced: a kernel variant in ~25 s for S=256 without a full build), `profl.sh BIN
NEFF FILE TAG [LEN] [IDX]` (real-data profile through ENGINE_PROF_*), `tools/gaps.py` (TensorE gaps per physical core
with the source lines around them), `tools/pe_iv.py` (matmul issue intervals, HAM), `tools/phases.py` (per time bin:
dominant PE source line + DMA GB/s per core), `tools/dbins.py`, `tools/win.py`, `evar.sh` / `mvar.sh` (variants of
pf_layer_256 / pf_multi15_256 relinked for e2e A/B), `cachekey.py` (cache file name of a variant: drop a NEFF built
by one.sh into target/neffcache). e2e A/B: `ab.py` with `--decode 1 --rounds 20` (prefill only) resolves ~0.1%
(A/A 1.0003); the cores were swapped once to check the core bias (none measurable for prefill this round).
**Build:** `build_neffs.py` no longer hashes itself into every key (a variant's key = its kernel module + common.py
(+ pattn, nkattn, decode.py for prefill) + the variant's argument shapes + `BUILD_KEY_VERSION`): adding a variant
does not rebuild everything. Any prefill.py edit rebuilds all prefill NEFFs; `pf_multi15_1024` alone takes ~11 min,
so a clean build is now ~25-30 min (8 jobs).

Host (from the rejected round 010, tokens unchanged): ENGINE_STEPTIME / ENGINE_PROF_* / ENGINE_SKIP /
ENGINE_PF_PATTERN / ENGINE_DEC_PATTERN diagnostics and the allocation-free JSON integer path. New: ENGINE_TIMING
(host-side prefill timing), ENGINE_NO_PF_FIRST / _LASTH / _MULTI (fall back to the round-009 execution sequence).

| change | measured | tokens |
|---|---|---|
| `pf_lasth_S` (S = 256, 1024, 2048, 4096): last layer + final norm + LM head + argmax in one NEFF (the tail's two output halves meet in column 1 of `hs`; `_lasth_head`) | S=256 0.95 + 1.68 -> 2.57 ms (steptime) | identical (ab) |
| S = 256 last layer: K/V for all tokens only (`_ns_last_kv`, keeps K^T / V tiles in SBUF), Q and attention for the last token only, decode style (`_ns_last_tokattn`: q GEMV, scores over S keys, keys > pos masked, softmax, P^T by identity matmuls, PV) | pf_lasth_256 2.57 -> 2.42 ms; e2e p256 ~-0.1 ms (the separate executions had overlapped part of it) | not bitwise (attention of the last token as decode computes it); ab tokens identical |
| `NS_XN2X` (S <= 256): each core scales its own h1 half by g2 as o_proj produces it and stores bf16(h1 * g2) to `hs` (what the other core's gate/up reads) instead of h1: the receiver loads it straight into xn2, no scaling after the exchange (the TensorE waited on the Scalar engine there) | pf_layer_256 profile 0.864 -> 0.855 ms | bitwise (same elementwise ops) |
| `NS_NB = 8` (was 4) for o_proj and down (`NS_NBD`): 8 output tiles accumulate the own K half before the other half is needed | 0.855 -> 0.8475 ms; e2e p256 +2.0% with the two rows above (ab 4 rounds vs round 009) | bitwise |
| `pf_first_S` (S <= 256): embedding gather (all S rows on each core, PE transposes, exact) fused into layer 0 (`_embed_full`, `xsb`) | pf_first_256 profile 0.871 ms vs layer 0.85 + pf_embed ~0.2 + a NEFF switch; e2e ~-0.1 ms | identical |
| `pf_multi15_256`: layers 1-15 and 16-30 as two executions of one 15-layer NEFF (`_pf_multi_body`: layer output into t0 / t1 scratch + core barrier). Alone: no gain (in-NEFF boundary ~20 us of x reload + norm, the same as a NEFF switch). With `PF_MULTI_OWN`: the next layer keeps its own half of x in SBUF (the previous layer's down output) and accumulates the first `MULTI_PRE = 6` QKV tiles (k_0, v_0, q_0..3) over the own K half while the other core's half is loaded | per layer 0.843 (e2e, separate NEFFs) -> 0.836 ms; **e2e p256 +1.4%** (ab 20 rounds, prefill only) | not bitwise (core 1's QKV fp32 order: own half first); 1 of 20 ab requests differs |
| `pf_multi15_1024`: the same for the S = 1024 pipe layer, without a barrier between layers (token split: each core's next-layer input is its own tokens, written and read by itself); q / K / V / attention scratch alternates between two sets so a core in layer j+1 never overwrites what the other core still reads in layer j (layer j+1's K/V barrier separates j and j+2) | **e2e p1024 +0.9%** vs round 009 (ab 20 rounds, both core orders) | bitwise (ab identical) |
| `NS_ATT2` (S <= 256): the last kv group's attention heads stored to `qs` (unused by these layers) with their own core barrier and load, so o_proj's first 28 K tiles do not wait for that store's latency | e2e p256 +0.2% (ab 20 rounds; noise level, kept: bitwise) | bitwise |

Result (ab.py, round-009 binary vs this engine, 20 rounds prefill-only, both core assignments): **p256 +3.2..3.3%,
p1024 +0.9%**, p4096 +-0; decode unchanged (4 rounds with 64-token decode: +0.2 / 0.0 / -0.1%).
dev-eval 001 (2 rounds, before NS_ATT2): PASSED, 0 mismatching steps, score 1.6581 (prefill 2.0066, decode 1.3701);
prefill speedups 2.408 / 2.321 / 1.446, engine prefill 28.55 / 104.15 / 419.6 ms.
**Final engine, dev-eval 002 (3 rounds, built by dev-eval): PASSED, 0 mismatching steps in every round, score 1.6635
(prefill 2.0138, decode 1.3742)**; prefill speedups 2.423 / 2.329 / 1.448 (round 009: 2.334 / 2.300 / 1.444), engine
prefill 28.4 / 104.1 / 419.4 ms (round 009: 29.3 / 105.1 / 420.0), decode 1433.7 / 1436.8 / 1500.9 ms (unchanged).
Verification of the final source: `same.py` vs the round-009 binary 40/40 identical (all shapes consistent);
`lab/longcheck.py` (200 + 3840, 4000 + 161, 4097 + 65, 4161 + 1, 3000 + 1 + 1099, 2049 + 20 + 2000, prompts
1/2/127/128/129/255/256/257/513/1000/1023/1024/1025/2047/2048/3333/4095/4096): every request returned all its tokens;
23/24 identical to round 009, 200 + 3840 differs from token 4 (S = 256 numerics: core 1's own-half-first QKV order).

### Tried and rejected this round (numbers)

- Decode MLP_SHIFT 2 / 3 at bucket 384 (the profile shows physical core 0's DMA idle 1.05 ms of 10.83 per
  grp16e_384 vs 0.23 ms on core 1, i.e. core 1 has more bytes): decode 0.9983 / 1.0035 (noise). Per-phase stream
  rates: core 0 305-317 GB/s, core 1 344-346 in every GEMV phase (a uniform ~10% asymmetry, as round 008 found), so
  MLP_SHIFT 4 stays.
- S = 256 layer start: rstd sums after the first QKV tile (`NS_RSTD_LATE`) and x in 8 pieces: e2e 0.9989 / 1.0004.
- Exchange loads on static-descriptor DMAs (`dge_mode.none`), on the Scalar queue (NS_XQ / NS_XQM / NS_XQH = 2,
  all or each): neutral to -1.5% (profile). The DMA engine of a `dma_copy` can only be chosen for hwdge (Sync /
  Scalar); none / swdge take no engine argument.
- sendrecv for the h1 exchange: gpsimd_dma sendrecv is limited to 512 bytes per partition (one [128, 256] bf16 tile)
  and **hung** (NRT_TIMEOUT) with 16 per-tile calls per layer; `dma_engine.dma` works but: raw h1 0.872 vs 0.864 ms;
  bf16(h1 * g2) in 4-tile pieces (+ partial sums) e2e 1.0037, 1-tile pieces 0.9937. Not kept.
- NS_NB 16: 0.900 ms; NS_NBD 4 (with NS_NB 8): -0.64% e2e; NS_NBG 3 / 4: neutral.
- PIPE_A2 (S = 1024: phase A of the first own chunk, K/V barrier, its attention spread over phase A of the second
  chunk, second barrier): layer 3.268 -> 3.325 ms (phase A slows from ~340% to ~295% PE in the profile: the
  attention's ACT / DVE stages convoy with RoPE / copies). Code kept, switched off.
- `pf_multi6_4096` (6 S = 4096 layers per NEFF, same scheme as 1024): p4096 -0.26% (421.5 vs 420.3 ms); not built
  (a 15-layer version would compile for ~40 min).
- MULTI_PRE 8 (adds k_1, v_1): -1.4%; 4: neutral.
- Last-layer tail with core 1 taking 1 / 3 / 1 of core 0's o_proj / gate-up / down tiles (`TAIL_S*`, bitwise):
  pf_lasth_256 2.41 ms either way (code kept, set to 0).
- Decode-style last-token attention for pf_lasth_tok (1024 / 4096, `LH_TOKATT_TOK`): 0.9999 / 1.0000.
- Host: the request's fixed cost is the runtime's first `nrta_execute_schedule` (~40-50 us); the token / position
  writes take ~1 us. Nothing to remove.
- The whole prefill in 2 executions (`pf_multi16e_S`: embedding + layers 0-15, `pf_multi15h_S`: layers 16-30 + last
  layer + head; S = 256 and 1024, code kept, `PF_EH = ()` so not built): p256 0.9994, p1024 1.0011 (ab 16 rounds).
  NEFF switches between queued executions cost nothing measurable here (round 010's 40-45 us switch cost is per
  execution when they are not queued back to back?). 2 of 32 ab requests differed in tokens.
- Gotcha: the host's pf_multi<G> binder must not match pf_multi15h / pf_multi16e (atoi("15h") = 15).

### Where the time is now (p256, 28.4 ms on core 51 alone)

pf_first 0.87 + 2 x pf_multi15 12.53 + pf_lasth 2.41 ms + ~0.2 ms client/protocol. In pf_multi15_256 a layer is 0.836
ms against 0.732 ms of matmul issue at 2.4 GHz: ~49 us of TensorE gaps per layer per core (o_proj waits for the other
core's attention heads ~7.5 us, gate/up ~9 us, down ring weights ~9 us, layer boundary ~5-9 us) plus ~35 us of slow
issue (HAM half-rate windows after gaps, small attention matmuls). The weight stream (218 MB per core per layer, ~650 us
at ~335 GB/s) is nearly as long as the matmuls, so any hiccup on either side shows.

### Next ideas (not measured)

- (measured instead: 2-NEFF prefill, no gain, above.)
- LM head split HSK 16 -> ~21 (core 0 finishes the head ~35 us after core 1 in pf_lasth_256; decode-wide change).
- S = 1024: the first own chunk's attention still runs alone (~40-50 us of half-idle TensorE per layer).
