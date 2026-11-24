"""Shared constants and small NKI helpers for the Llama-3.1-8B engine (Trainium2, LNC=2).

Layouts (all chosen so no activation transposes are needed between decode GEMVs):
  column vector of length K: SBUF [128, K/128], element k at [k % 128, k // 128].
  weight tile (nt, kt) of W [N, K] (row = output): [128 k, 128 n] with
      tile[p, j] = W[nt*128 + j, kt*128 + p]
  a per-core weight run is [128, NT*KT*128] with tile (nt, kt) at columns (nt*KT + kt)*128.
"""
import nki.language as nl
import nki.isa as nisa

H = 4096
HT = H // 128          # 32 hidden tiles
NQ = 32
NKV = 8
HD = 128
I = 14336
IT = I // 128          # 112 intermediate tiles
ITC = IT // 2          # 56 per core
VOCAB = 128256
VT = VOCAB // 128      # 1002 vocab tiles
VTC = VT // 2          # 501 per core
HSK = 16               # LM head: core 0 takes VTC - HSK vocab tiles, core 1 VTC + HSK (core 0 streams slower)
VTCMAX = VTC + HSK
TMAX = 4224            # KV cache positions (33 x 128)
EPS = 1e-5
QKV_NT = 24            # per core: 16 q heads + 4 k heads + 4 v heads
GU_G = 8               # intermediate tiles per gate/up group

BF16 = nl.bfloat16
F32 = nl.float32


def wdma(dst, src):
    """Weight stream: hardware DGE issued from the Sync engine (computes nothing)."""
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def gemv_cols(w2d, x_sb, NT, KT, ps, slab_nt=1):
    """ps[:, nt] (fp32 PSUM, [128, NT]) = sum_kt tile(nt, kt)^T @ x_sb[:, kt]  for nt < NT.

    w2d: HBM [128, NT*KT*128] (this core's run), x_sb: SBUF [128, KT] bf16."""
    W = KT * 128
    for s in range(NT // slab_nt):
        wt = nl.ndarray((128, slab_nt * W), dtype=BF16, buffer=nl.sbuf)
        wdma(wt, w2d[:, s * slab_nt * W:(s + 1) * slab_nt * W])
        for a in range(slab_nt):
            nt = s * slab_nt + a
            for kt in range(KT):
                nisa.nc_matmul(dst=ps[:, nt:nt + 1], stationary=wt[:, (a * KT + kt) * 128:(a * KT + kt + 1) * 128],
                               moving=x_sb[:, kt:kt + 1])


def rmsnorm_col(x_bf, g_bf, ones_f):
    """RMSNorm on a column vector [128, HT] bf16 as NxD CustomRMSNorm: bf16(x * rstd * g) in fp32."""
    sq = nl.ndarray((128, HT), dtype=F32, buffer=nl.sbuf)
    ss = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=x_bf, reduce_op=nl.add, reduce_res=ss,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    tot = nl.ndarray((128, 1), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(dst=tot, stationary=ones_f, moving=ss)
    t = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=tot, op0=nl.multiply, operand0=1.0 / H, op1=nl.add, operand1=EPS)
    rstd = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=t, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    xn = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=xn, data=x_bf, op0=nl.multiply, operand0=rstd, op1=nl.multiply, operand1=g_bf)
    return xn


def exchange_add(part_f, pipe):
    """Sum a [128, n] fp32 partial over the two physical cores (same result bytes on both)."""
    pid = nl.program_id(0)
    n = part_f.shape[1]
    other = nl.ndarray((128, n), dtype=F32, buffer=nl.sbuf)
    nisa.sendrecv(src=part_f, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=pipe,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    tot = nl.ndarray((128, n), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot, data1=part_f, data2=other, op=nl.add)
    return tot


RING_W = 4096
RW = RING_W               # ring slot width (columns = 32 weight tiles, 8 KB per partition)


def _ring_slots(NT, KT, rw):
    """The slots of a ring GEMV in order: (nt, k0, n, nt2) as _ring_slot takes them."""
    tps = rw // 128
    out = []
    if KT <= tps:
        per = tps // KT
        ns = (NT + per - 1) // per
        for s in range(ns):
            nt0 = s * per
            out.append((nt0, 0, KT, min(NT, nt0 + per)))
        return out
    nsub = (KT + tps - 1) // tps
    for nt in range(NT):
        for j in range(nsub):
            k0 = j * tps
            out.append((nt, k0, min(tps, KT - k0), nt + 1))
    return out


def _slot_dma(w2d, ring, slot, nt, KT, k0, n, nt2):
    col0 = (nt * KT + k0) * 128
    ncols = (n + (nt2 - nt - 1) * KT) * 128
    wdma(ring[:, slot, 0:ncols], w2d[:, col0:col0 + ncols])


def _ring_slot(w2d, x_sb, ps, ring, slot, nt, KT, k0, n, nt2, dma=True):
    """One ring slot: tiles (nt, k0 .. k0+n-1) and, if nt2 > nt, all KT tiles of nt+1 .. nt2-1 after them
    (contiguous in the run), then their matmuls in the gemv_cols order. dma=False: the slot's DMA was issued
    earlier (gemv_ring_prefetch)."""
    if dma:
        _slot_dma(w2d, ring, slot, nt, KT, k0, n, nt2)
    for kt in range(n):
        nisa.nc_matmul(dst=ps[:, nt:nt + 1], stationary=ring[:, slot, kt * 128:(kt + 1) * 128],
                       moving=x_sb[:, k0 + kt:k0 + kt + 1])
    for j in range(nt2 - nt - 1):
        for kt in range(KT):
            c = n + j * KT + kt
            nisa.nc_matmul(dst=ps[:, nt + 1 + j:nt + 2 + j], stationary=ring[:, slot, c * 128:(c + 1) * 128],
                           moving=x_sb[:, kt:kt + 1])


def gemv_ring_prefetch(w2d, NT, KT, ring, base, NB, rw, npre):
    """Issue the weight DMAs of the first npre slots of a later gemv_ring(..., base, pre=npre) now (in program
    order), so the compiler schedules them before the instructions in between (e.g. decode attention)."""
    sl = _ring_slots(NT, KT, rw)
    npre = min(npre, len(sl), NB)
    for i in range(npre):
        nt, k0, n, nt2 = sl[i]
        _slot_dma(w2d, ring, (base + i) % NB, nt, KT, k0, n, nt2)
    return npre


def gemv_ring(w2d, x_sb, NT, KT, ps, ring, base, NB, rw=RW, pre=0):
    """gemv_cols with the weight tiles streamed through an explicit ring of NB slots [128, NB, RW] (same matmuls,
    same order). base: running slot counter (trace time); returns the new counter. A slot is reused only after
    the matmuls reading it, so the weight DMAs can run up to NB slots ahead of the PE across GEMVs and layers.
    KT <= RW/128: as many whole output tiles per slot as fit; else each output tile over several slots.
    pre: the DMAs of the first `pre` slots were issued by gemv_ring_prefetch."""
    sl = _ring_slots(NT, KT, rw)
    for i in range(len(sl)):
        nt, k0, n, nt2 = sl[i]
        _ring_slot(w2d, x_sb, ps, ring, (base + i) % NB, nt, KT, k0, n, nt2, i >= pre)
    return base + len(sl)
