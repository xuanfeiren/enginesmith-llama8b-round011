"""Decode-step NEFFs: embedding gather, one transformer layer (run 32x on per-layer slices), head + argmax.

Each physical core (pid 0/1) computes half of every GEMV:
  QKV:   q heads 16c..16c+15, kv heads 4c..4c+3 (N split)
  o:     K split over its 16 heads -> fp32 partial, summed over cores with sendrecv
  gate/up: N split over intermediate tiles 56c..56c+55, down: K split over the same tiles
  (round 006, MLP_SHIFT = 4: core 0's last 4 intermediate tiles -- gate/up and their down K tiles, read from core
  0's weight runs -- are computed by core 1, which adds their down partial into its own before exchange 2; core 0
  is the slower core, decode +2.3..3.6%. Not bitwise vs round 005: fp32 partial-sum grouping of the down proj.)
"""
import nki
import nki.language as nl
import nki.isa as nisa

from common import (H, HT, HD, IT, ITC, VT, VTC, HSK, VTCMAX, TMAX, QKV_NT, GU_G, BF16, F32, RW,
                    wdma, gemv_cols, gemv_ring, gemv_ring_prefetch, rmsnorm_col, exchange_add)


def gemv_ring_rows(wt, c, x_sb, NT, KT, kofs, KS, ps, ring, base, NB, rw=RW):
    """ps[:, nt] = sum_{kt < KT} tile(nt, kofs + kt) @ x_sb[:, kt] for nt < NT, the tiles taken from the run wt[c]
    (wt: HBM [2, 128, NT*KS*128], row nt's KS tiles contiguous) through the ring. kofs = 0, KT = KS is gemv_ring.
    KT <= ring tiles: several rows per slot (one strided DMA); else each row over several slots."""
    tps = rw // 128
    C = NT * KS * 128
    if KT <= tps:
        per = min(tps // KT, NT)
        s = 0
        for nt0 in range(0, NT, per):
            nr = min(per, NT - nt0)
            slot = (base + s) % NB
            if nr == 1 or KT == KS:
                col0 = (nt0 * KS + kofs) * 128
                ncols = ((nr - 1) * KS + KT) * 128 if KT == KS else KT * 128
                wdma(ring[:, slot, 0:ncols], wt[c, :, col0:col0 + ncols])
            else:
                nisa.dma_copy(dst=ring.ap(pattern=[[NB * rw, 128], [KT * 128, nr], [1, KT * 128]], offset=slot * rw),
                              src=wt.ap(pattern=[[C, 128], [KS * 128, nr], [1, KT * 128]],
                                        offset=c * 128 * C + (nt0 * KS + kofs) * 128),
                              dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            for r in range(nr):
                for kt in range(KT):
                    cc = r * KT + kt
                    nisa.nc_matmul(dst=ps[:, nt0 + r:nt0 + r + 1], stationary=ring[:, slot, cc * 128:(cc + 1) * 128],
                                   moving=x_sb[:, kt:kt + 1])
            s += 1
        return base + s
    s = 0
    for nt in range(NT):
        for k0 in range(0, KT, tps):
            n = min(tps, KT - k0)
            slot = (base + s) % NB
            col0 = (nt * KS + kofs + k0) * 128
            wdma(ring[:, slot, 0:n * 128], wt[c, :, col0:col0 + n * 128])
            for kt in range(n):
                nisa.nc_matmul(dst=ps[:, nt:nt + 1], stationary=ring[:, slot, kt * 128:(kt + 1) * 128],
                               moving=x_sb[:, k0 + kt:k0 + kt + 1])
            s += 1
    return base + s


RING_NB = 10   # weight ring slots per core (0: the compiler's own per-slab buffers)
KV_ENGINE = nisa.engine.scalar
RING_W16_MAXB = 2048
RING16_NB = 6
HEAD_RING_NB = 4      # LM head: weight ring slots of HEAD_RW columns (0: 8 KB slabs, compiler buffers)
HEAD_RW = 16384


def _ring_cfg(B):
    """(slots, slot width in columns) of the decode weight ring for attention bucket B: 16 KB slots while the
    KV tiles leave room in SBUF (measured end to end: +0.1% / +0.7% decode at B 384 / 1152, -0.85% at 4224),
    8 KB slots above."""
    if B <= RING_W16_MAXB:
        return RING16_NB, 8192
    return RING_NB, RW


def kvdma(dst, src, B, eng=None):
    if eng == 'sw':
        nisa.dma_copy(dst=dst, src=src)
        return
    """KV-cache load on the Sync engine's hardware DGE queue (measured: b384 group profile 2.920 -> 2.899 ms per 4
    layers, decode +0.4..1.0% end to end; static descriptors (dge_mode.none) no different at b4224)."""
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=KV_ENGINE if eng is None else eng)


KVW_MODE = 1   # K/V cache writes: 0 default (software DGE on GpSimd, ~10 us latency), 1 hwdge on KVW_ENGINE
KVW_ENGINE = nisa.engine.scalar
VROW = 1
MLP_MOVE = 0   # >0: core 0's last gate/up tiles computed by core 1 and sent over (decode +3.5% when it runs, but the gpsimd sendrecv hangs intermittently; standard-DMA sendrecv is safe but 3.5% slower). Keep 0.
MV_DMA = nisa.dma_engine.dma   # moved tiles: standard DMA sendrecv with CoreBarrier sync (gpsimd_dma hung)
MLP_SHIFT = 4   # core 0's last MLP_SHIFT intermediate tiles (gate/up and their down K tiles) computed by core 1;
               # the moved down partial is added into core 1's partial before exchange 2 (no extra sync)
O_PRE = 6   # o_proj ring slots prefetched before the attention (0: off)
KEARLY = 1
VPATCH = 1
KPATCH_ENGINE = nisa.engine.vector
KL_ENGINE = nisa.engine.sync
VW_ENGINE = nisa.engine.scalar
VL_ENGINE = nisa.engine.sync


def kvwdma(dst, src, eng=None):
    if eng == 'sw':
        nisa.dma_copy(dst=dst, src=src)
        return
    if KVW_MODE:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=KVW_ENGINE if eng is None else eng)
    else:
        nisa.dma_copy(dst=dst, src=src)


def _load_pos(pos):
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    return p_sb, preg


def _xbuf():
    """Send / receive buffers of the moved MLP tiles (MLP_MOVE), one pair per NEFF on both cores."""
    if not MLP_MOVE:
        return None
    xs = nl.ndarray((128, MLP_MOVE), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=xs, value=0.0)
    xr = nl.ndarray((128, MLP_MOVE), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=xr, value=0.0)
    return (xs, xr)


def layer_body(h, pos, cosT, sinT, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, B, DEBUG=0):
    p_sb, preg = _load_pos(pos)
    ones_f = nl.ndarray((128, 128), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    cst = nl.ndarray((128, 256), dtype=BF16, buffer=nl.sbuf)   # [rperm | identity]
    nisa.dma_copy(dst=cst, src=consts)
    h_bf = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=h_bf, src=h)
    if RING_NB and not DEBUG:
        nb, rw = _ring_cfg(B)
        ring = nl.ndarray((128, nb, rw), dtype=BF16, buffer=nl.sbuf)
        r, rb = layer_sb(h_bf, pos, preg, ones_f, cst, cosT, sinT, g1, wqkv, wo, g2, wgu, wd, kc, vc, B, DEBUG, ring, 0,
                         nb, rw, _xbuf())
    else:
        r = layer_sb(h_bf, pos, preg, ones_f, cst, cosT, sinT, g1, wqkv, wo, g2, wgu, wd, kc, vc, B, DEBUG)
    if DEBUG:
        return r
    h_out = nl.ndarray((128, HT), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=h_out, src=r)
    return h_out


def layer_sb(h_bf, pos, preg, ones_f, cst, cosT, sinT, g1, wqkv, wo, g2, wgu, wd, kc, vc, B, DEBUG=0, ring=None, rb=0,
             NB=0, rw=RW, xbuf=None):
    """One decode layer on the SBUF hidden column h_bf [128, HT] bf16; returns the new hidden column (SBUF)."""
    pid = nl.program_id(0)
    NTT = B // 128
    NCH = (B + 511) // 512
    g1_sb = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=g1_sb, src=g1)
    g2_sb = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=g2_sb, src=g2)
    cos_c = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_c, src=cosT.ap(pattern=[[1, 128], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
    sin_c = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sin_c, src=sinT.ap(pattern=[[1, 128], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))

    if KEARLY:   # K / V cache [0, B) loaded before this token's K / V are written (stale at pos, patched in SBUF)
        p_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=p_i, src=pos)
        p_u = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=p_u, src=p_i)
        kt_sb = []
        for g in range(4):
            t = nl.ndarray((128, B), dtype=BF16, buffer=nl.sbuf)
            kvdma(t, kc[pid * 4 + g, :, 0:B], B, KL_ENGINE)
            kt_sb.append(t)
        if VPATCH:
            v_sb = nl.ndarray((128, NTT, 512), dtype=BF16, buffer=nl.sbuf)
            kvdma(v_sb, vc.ap(pattern=[[1024, 128], [128 * 1024, NTT], [1, 512]], offset=pid * 512), B, VL_ENGINE)
            # tile / row of pos: i5 = pos >> 7 (a scalar_offset counts in units of the indexed dim of the tensor's
            # shape: tile index of v_sb [128, NTT, 512]), r = pos & 127
            i5 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=i5, data=p_i, op0=nl.right_shift, operand0=7)
            i5u = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=i5u, src=i5)
            pb_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pb_i, src=pos.ap(pattern=[[0, 128], [1, 1]], offset=0))
            r_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=r_i, data=pb_i, op0=nl.bitwise_and, operand0=127)
            r_f = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=r_f, src=r_i)
            ip = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)       # partition index
            nisa.iota(dst=ip, pattern=[[0, 1]], offset=0, channel_multiplier=1)
            ome = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)      # 1 - (p == r)
            nisa.tensor_scalar(dst=ome, data=ip, op0=nl.not_equal, operand0=r_f)
            ifr = nl.ndarray((128, 128), dtype=F32, buffer=nl.sbuf)    # free index
            nisa.iota(dst=ifr, pattern=[[1, 128]], offset=0, channel_multiplier=0)
            erow = nl.ndarray((128, 128), dtype=F32, buffer=nl.sbuf)   # erow[d, p] = (p == r)
            nisa.tensor_scalar(dst=erow, data=ifr, op0=nl.equal, operand0=r_f)

    # ---- input norm + QKV
    xn = rmsnorm_col(h_bf, g1_sb, ones_f)
    ps_qkv = nl.ndarray((128, QKV_NT), dtype=F32, buffer=nl.psum)
    if ring is not None:
        rb = gemv_ring(wqkv[pid, :, :], xn, QKV_NT, HT, ps_qkv, ring, rb, NB, rw)
    else:
        gemv_cols(wqkv[pid, :, :], xn, QKV_NT, HT, ps_qkv)
    qkv = nl.ndarray((128, QKV_NT), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qkv, src=ps_qkv)
    opre = 0
    if ring is not None and O_PRE:   # o_proj weight slots issued before the attention (program order)
        opre = gemv_ring_prefetch(wo[pid, :, :], HT, 16, ring, rb, NB, rw, O_PRE)

    if KEARLY:   # this token's v: one 256-byte row per head to the cache (no one waits for it in this step)
        pvr = nl.ndarray((4, 128), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=pvr, stationary=qkv[:, 20:24], moving=cst[:, 128:256])
        vrow = nl.ndarray((4, 128), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vrow, src=pvr)
        kvwdma(vc.ap(pattern=[[128, 4], [1, 128]], offset=pid * 512, scalar_offset=preg, indirect_dim=0), vrow,
               VW_ENGINE)
        if VPATCH:   # row r of tile pos//128 of the loaded V := v (exact: x*1 + 0, x*0 + v)
            vf = nl.ndarray((128, 4), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=vf, src=qkv[:, 20:24])
            pvp = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)
            for j in range(4):
                sj = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)   # sj[d, p] = v_j[d] * (p == r)
                nisa.tensor_scalar(dst=sj, data=erow, op0=nl.multiply, operand0=vf[:, j:j + 1])
                nisa.nc_matmul(dst=pvp[:, j * 128:(j + 1) * 128], stationary=sj, moving=cst[:, 128:256])
            vt = nl.ndarray((128, 512), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=vt, src=v_sb.ap(pattern=[[NTT * 512, 128], [1, 512]], offset=0, scalar_offset=i5u,
                                                  indirect_dim=1))
            vt2 = nl.ndarray((128, 512), dtype=BF16, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=vt2, data=vt, op0=nl.multiply, operand0=ome, op1=nl.add, operand1=pvp)
            nisa.tensor_copy(dst=v_sb.ap(pattern=[[NTT * 512, 128], [1, 512]], offset=0, scalar_offset=i5u,
                                         indirect_dim=1), src=vt2)
        else:
            v_sb = nl.ndarray((128, NTT, 512), dtype=BF16, buffer=nl.sbuf)
            kvdma(v_sb, vc.ap(pattern=[[1024, 128], [128 * 1024, NTT], [1, 512]], offset=pid * 512), B, VL_ENGINE)

    # ---- RoPE on q (cols 0..15) and k (16..19): x*cos + perm(x)*sin_signed
    ps_rot = nl.ndarray((128, 20), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps_rot, stationary=cst[:, 0:128], moving=qkv[:, 0:20])
    # bf16 products and sum, as NxD (cos/sin tables hold bf16-rounded values)
    a_ = nl.ndarray((128, 20), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a_, data=qkv[:, 0:20], op0=nl.multiply, operand0=cos_c)
    b_ = nl.ndarray((128, 20), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=b_, data=ps_rot, op0=nl.multiply, operand0=sin_c)
    qk = nl.ndarray((128, 20), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=qk, data1=a_, data2=b_, op=nl.add)

    # ---- write this token's K (transposed cache [8, 128, TMAX]) and V ([TMAX, 1024]) at pos
    kvwdma(kc.ap(pattern=[[TMAX, 128], [128 * TMAX, 4]], offset=pid * 4 * 128 * TMAX,
                 scalar_offset=preg, indirect_dim=2), qk[:, 16:20])
    if KEARLY:
        for g in range(4):   # this token's k into column pos of the loaded K^T (exact copy)
            nisa.tensor_copy(dst=kt_sb[g].ap(pattern=[[B, 128], [1, 1]], offset=0, scalar_offset=p_u, indirect_dim=1),
                             src=qk[:, 16 + g:17 + g], engine=KPATCH_ENGINE)
    else:
        if VROW:   # v as 4 rows of 128 (PE transpose, exact): 4 contiguous 256-byte writes instead of 512 2-byte ones
            pvr = nl.ndarray((4, 128), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(dst=pvr, stationary=qkv[:, 20:24], moving=cst[:, 128:256])
            vrow = nl.ndarray((4, 128), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=vrow, src=pvr)
            kvwdma(vc.ap(pattern=[[128, 4], [1, 128]], offset=pid * 512, scalar_offset=preg, indirect_dim=0), vrow)
        else:
            kvwdma(vc.ap(pattern=[[1, 128], [128, 4]], offset=pid * 512, scalar_offset=preg, indirect_dim=0),
                   qkv[:, 20:24])
        # ---- cache loads (ordered after the writes above by the compiler: same tensors)
        kt_sb = []
        for g in range(4):
            t = nl.ndarray((128, B), dtype=BF16, buffer=nl.sbuf)
            kvdma(t, kc[pid * 4 + g, :, 0:B], B)
            kt_sb.append(t)
        v_sb = nl.ndarray((128, NTT, 512), dtype=BF16, buffer=nl.sbuf)
        kvdma(v_sb, vc.ap(pattern=[[1024, 128], [128 * 1024, NTT], [1, 512]], offset=pid * 512), B)

    # ---- attention over cache positions [0, B), keys > pos masked
    # score rows = 16 local heads; per kv group g a zero-padded q stationary so groups accumulate exactly
    qm = []
    for g in range(4):
        t = nl.ndarray((128, 16), dtype=BF16, buffer=nl.sbuf)
        nisa.memset(dst=t, value=0.0)
        nisa.tensor_copy(dst=t[:, 4 * g:4 * g + 4], src=qk[:, 4 * g:4 * g + 4])
        qm.append(t)

    posf_i = nl.ndarray((16, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=posf_i, src=pos.ap(pattern=[[0, 16], [1, 1]], offset=0))
    posf = nl.ndarray((16, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=posf, src=posf_i)
    iot = nl.ndarray((16, B), dtype=F32, buffer=nl.sbuf)
    nisa.iota(dst=iot, pattern=[[1, B]], offset=0, channel_multiplier=0)
    mb = nl.ndarray((16, B), dtype=F32, buffer=nl.sbuf)   # -30000 where t > pos
    nisa.tensor_scalar(dst=mb, data=iot, op0=nl.greater, operand0=posf, op1=nl.multiply, operand1=-30000.0)
    sc = nl.ndarray((16, B), dtype=F32, buffer=nl.sbuf)
    scale = 1.0 / (HD ** 0.5)
    for j in range(NCH):
        w = min(512, B - j * 512)
        ps = nl.ndarray((16, 512), dtype=F32, buffer=nl.psum)
        for g in range(4):
            nisa.nc_matmul(dst=ps[:, 0:w], stationary=qm[g], moving=kt_sb[g][:, j * 512:j * 512 + w])
        s1 = nl.ndarray((16, 512), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s1[:, 0:w], src=ps[:, 0:w])
        s2 = nl.ndarray((16, 512), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=s2[:, 0:w], data=s1[:, 0:w], op0=nl.multiply, operand0=scale)
        nisa.tensor_tensor(dst=sc[:, j * 512:j * 512 + w], data1=s2[:, 0:w], data2=mb[:, j * 512:j * 512 + w],
                           op=nl.add)
    mx = nl.ndarray((16, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.maximum, data=sc, axis=(1,))
    nmx = nl.ndarray((16, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0)
    e = nl.ndarray((16, B), dtype=F32, buffer=nl.sbuf)
    ssum = nl.ndarray((16, 1), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=e, op=nl.exp, data=sc, bias=nmx, scale=1.0, reduce_op=nl.add, reduce_res=ssum,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rinv = nl.ndarray((16, 1), dtype=F32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rinv, data=ssum)
    pb = nl.ndarray((16, B), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pb, data=e, op0=nl.multiply, operand0=rinv)

    # P^T tiles [128 t, 16] via identity matmuls (exact)
    pT = nl.ndarray((128, NTT, 16), dtype=BF16, buffer=nl.sbuf)
    for i0 in range(0, NTT, 32):
        n = min(32, NTT - i0)
        pst = nl.ndarray((128, 32, 16), dtype=F32, buffer=nl.psum)
        for i in range(n):
            nisa.nc_matmul(dst=pst[:, i, :], stationary=pb[:, (i0 + i) * 128:(i0 + i + 1) * 128],
                           moving=cst[0:16, 128:144])
        nisa.tensor_copy(dst=pT[:, i0:i0 + n, :], src=pst[:, 0:n, :])

    attn = nl.ndarray((128, 16), dtype=BF16, buffer=nl.sbuf)
    for g in range(4):
        ps_o = nl.ndarray((128, 4), dtype=F32, buffer=nl.psum)
        for i in range(NTT):
            nisa.nc_matmul(dst=ps_o, stationary=v_sb[:, i, g * 128:(g + 1) * 128],
                           moving=pT[:, i, 4 * g:4 * g + 4])
        nisa.tensor_copy(dst=attn[:, 4 * g:4 * g + 4], src=ps_o)

    # ---- o_proj (K split) + residual
    ps_op = nl.ndarray((128, HT), dtype=F32, buffer=nl.psum)
    if ring is not None:
        rb = gemv_ring(wo[pid, :, :], attn, HT, 16, ps_op, ring, rb, NB, rw, opre)
    else:
        gemv_cols(wo[pid, :, :], attn, HT, 16, ps_op, slab_nt=2)
    part = nl.ndarray((128, HT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=part, src=ps_op)
    tot = exchange_add(part, 0)
    ob = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ob, src=tot)
    h1 = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h1, data1=h_bf, data2=ob, op=nl.add)

    # ---- MLP
    xn2 = rmsnorm_col(h1, g2_sb, ones_f)
    mlp_in = nl.ndarray((128, ITC), dtype=BF16, buffer=nl.sbuf)
    wgu_c = wgu[pid, :, :]
    WG = HT * 128
    NGI = ITC // GU_G
    MV = MLP_MOVE if ring is not None else 0
    if MV:   # sendrecv writes the peer's dst at the same SBUF address, possibly before the peer reaches its own
        # sendrecv: xs / xr are allocated once per NEFF (before the layers), never reused for anything else
        xs = xbuf[0]
        xr = xbuf[1]
    if MV and pid == 1:   # core 0's last MV intermediate tiles (same instructions as core 0 would issue), sent to it
        g0 = (NGI - 1) * 2 * GU_G + GU_G - MV
        u0 = (NGI - 1) * 2 * GU_G + 2 * GU_G - MV
        wo0 = wgu[0, :, :]
        ps_x = nl.ndarray((128, 2 * MV), dtype=F32, buffer=nl.psum)
        rb = gemv_ring(wo0[:, g0 * WG:(g0 + MV) * WG], xn2, MV, HT, ps_x[:, 0:MV], ring, rb, NB, rw)
        rb = gemv_ring(wo0[:, u0 * WG:(u0 + MV) * WG], xn2, MV, HT, ps_x[:, MV:2 * MV], ring, rb, NB, rw)
        gx = nl.ndarray((128, MV), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gx, src=ps_x[:, 0:MV])
        ux = nl.ndarray((128, MV), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ux, src=ps_x[:, MV:2 * MV])
        ax = nl.ndarray((128, MV), dtype=BF16, buffer=nl.sbuf)
        nisa.activation(dst=ax, op=nl.silu, data=gx)
        mx = nl.ndarray((128, MV), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mx, data1=ax, data2=ux, op=nl.multiply)
        nisa.tensor_copy(dst=xs, src=mx)
        nisa.sendrecv(src=xs, dst=xr, send_to_rank=0, recv_from_rank=0, pipe_id=2,
                      dma_engine=MV_DMA)
    SH = MLP_SHIFT if (ring is not None and not MV) else 0
    for gi in range(NGI):
        ng = GU_G - (MV + SH) if ((MV or SH) and pid == 0 and gi == NGI - 1) else GU_G
        ps_gu = nl.ndarray((128, 2 * GU_G), dtype=F32, buffer=nl.psum)
        base = gi * 2 * GU_G
        if ring is not None and ng < GU_G:   # gate tiles 0..ng-1 and up tiles 0..ng-1 of the group
            rb = gemv_ring(wgu_c[:, base * WG:(base + ng) * WG], xn2, ng, HT, ps_gu[:, 0:ng], ring, rb, NB, rw)
            rb = gemv_ring(wgu_c[:, (base + GU_G) * WG:(base + GU_G + ng) * WG], xn2, ng, HT, ps_gu[:, GU_G:GU_G + ng],
                           ring, rb, NB, rw)
        elif ring is not None:
            rb = gemv_ring(wgu_c[:, gi * 2 * GU_G * HT * 128:(gi + 1) * 2 * GU_G * HT * 128], xn2, 2 * GU_G, HT, ps_gu,
                           ring, rb, NB, rw)
        else:
            gemv_cols(wgu_c[:, gi * 2 * GU_G * HT * 128:(gi + 1) * 2 * GU_G * HT * 128], xn2, 2 * GU_G, HT, ps_gu)
        gb = nl.ndarray((128, GU_G), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gb[:, 0:ng], src=ps_gu[:, 0:ng])
        ub = nl.ndarray((128, GU_G), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ub[:, 0:ng], src=ps_gu[:, GU_G:GU_G + ng])
        ab = nl.ndarray((128, GU_G), dtype=BF16, buffer=nl.sbuf)
        nisa.activation(dst=ab[:, 0:ng], op=nl.silu, data=gb[:, 0:ng])
        nisa.tensor_tensor(dst=mlp_in[:, gi * GU_G:gi * GU_G + ng], data1=ab[:, 0:ng], data2=ub[:, 0:ng],
                           op=nl.multiply)
    if MV and pid == 0:   # receive the last MV tiles from core 1
        nisa.sendrecv(src=xs, dst=xr, send_to_rank=1, recv_from_rank=1, pipe_id=2,
                      dma_engine=MV_DMA)
        nisa.tensor_copy(dst=mlp_in[:, ITC - MV:ITC], src=xr)
    if SH and pid == 1:   # core 0's last SH gate/up tiles (same instructions as core 0 would issue)
        g0 = (NGI - 1) * 2 * GU_G + GU_G - SH
        u0 = (NGI - 1) * 2 * GU_G + 2 * GU_G - SH
        wo0 = wgu[0, :, :]
        ps_x = nl.ndarray((128, 2 * SH), dtype=F32, buffer=nl.psum)
        rb = gemv_ring(wo0[:, g0 * WG:(g0 + SH) * WG], xn2, SH, HT, ps_x[:, 0:SH], ring, rb, NB, rw)
        rb = gemv_ring(wo0[:, u0 * WG:(u0 + SH) * WG], xn2, SH, HT, ps_x[:, SH:2 * SH], ring, rb, NB, rw)
        gx = nl.ndarray((128, SH), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gx, src=ps_x[:, 0:SH])
        ux = nl.ndarray((128, SH), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ux, src=ps_x[:, SH:2 * SH])
        ax = nl.ndarray((128, SH), dtype=BF16, buffer=nl.sbuf)
        nisa.activation(dst=ax, op=nl.silu, data=gx)
        mxs = nl.ndarray((128, SH), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mxs, data1=ax, data2=ux, op=nl.multiply)
    ps_d = nl.ndarray((128, HT), dtype=F32, buffer=nl.psum)
    if SH and pid == 0:   # down K-split over this core's first ITC - SH tiles
        rb = gemv_ring_rows(wd, 0, mlp_in, HT, ITC - SH, 0, ITC, ps_d, ring, rb, NB, rw)
    elif SH:              # own tiles, then core 0's last SH tiles into a second partial
        rb = gemv_ring(wd[pid, :, :], mlp_in, HT, ITC, ps_d, ring, rb, NB, rw)
        ps_m = nl.ndarray((128, HT), dtype=F32, buffer=nl.psum)
        rb = gemv_ring_rows(wd, 0, mxs, HT, SH, ITC - SH, ITC, ps_m, ring, rb, NB, rw)
    elif ring is not None:
        rb = gemv_ring(wd[pid, :, :], mlp_in, HT, ITC, ps_d, ring, rb, NB, rw)
    else:
        gemv_cols(wd[pid, :, :], mlp_in, HT, ITC, ps_d)
    part2 = nl.ndarray((128, HT + MV), dtype=F32, buffer=nl.sbuf)
    if SH and pid == 1:   # (own + moved): exchange_add then gives P0 + (P1 + M) on both cores (fp32 add commutes)
        nisa.tensor_copy(dst=part2[:, 0:HT], src=ps_d)
        nisa.tensor_tensor(dst=part2[:, 0:HT], data1=part2[:, 0:HT], data2=ps_m, op=nl.add)
    else:
        nisa.tensor_copy(dst=part2[:, 0:HT], src=ps_d)
    if MV:   # the moved tiles' exchange (pipe 2) ordered before exchange 2 on core 1: its received block rides along
        nisa.tensor_scalar(dst=part2[:, HT:HT + MV], data=xr, op0=nl.add, operand0=xs[:, 0:1])
    tot2 = exchange_add(part2, 1)
    db = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=db, src=tot2[:, 0:HT])
    h2 = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h2, data1=h1, data2=db, op=nl.add)

    if DEBUG:
        h_out = nl.ndarray((128, HT), dtype=BF16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=h_out, src=h2)
        dbg = nl.ndarray((2, 128, 320), dtype=F32, buffer=nl.shared_hbm)
        dsb = nl.ndarray((128, 320), dtype=F32, buffer=nl.sbuf)
        nisa.memset(dst=dsb, value=0.0)
        nisa.tensor_copy(dst=dsb[:, 0:32], src=xn)
        nisa.tensor_copy(dst=dsb[:, 32:56], src=qkv)
        nisa.tensor_copy(dst=dsb[:, 56:76], src=qk)
        nisa.tensor_copy(dst=dsb[:, 76:92], src=attn)
        nisa.tensor_copy(dst=dsb[:, 92:124], src=h1)
        nisa.tensor_copy(dst=dsb[:, 124:156], src=xn2)
        nisa.tensor_copy(dst=dsb[:, 156:212], src=mlp_in)
        nisa.tensor_copy(dst=dsb[:, 212:244], src=part)
        nisa.tensor_copy(dst=dsb[:, 244:276], src=tot)
        nisa.dma_copy(dst=dbg[pid, :, :], src=dsb)
        dbg2 = nl.ndarray((2, 16, 2 * B), dtype=F32, buffer=nl.shared_hbm)
        d2 = nl.ndarray((16, 2 * B), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=d2[:, 0:B], src=sc)
        nisa.tensor_copy(dst=d2[:, B:2 * B], src=pb)
        nisa.dma_copy(dst=dbg2[pid, :, :], src=d2)
        dbg3 = nl.ndarray((2, 128, 512 + 16 * NTT), dtype=F32, buffer=nl.shared_hbm)
        d3 = nl.ndarray((128, 512 + 16 * NTT), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=d3[:, 0:512], src=v_sb[:, 2, :])
        nisa.tensor_copy(dst=d3[:, 512:512 + 16 * NTT], src=pT)
        nisa.dma_copy(dst=dbg3[pid, :, :], src=d3)
        return h_out, dbg, dbg2, dbg3
    if ring is not None:
        return h2, rb
    return h2


@nki.jit
def dec_layer(h, pos, cosT, sinT, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, B=512, DEBUG=0):
    return layer_body(h, pos, cosT, sinT, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, B, DEBUG)


@nki.jit
def dec_embed(toks, pos, embed):
    """h = embed[toks[pos]] in column layout."""
    p_sb, preg = _load_pos(pos)
    t_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t_sb, src=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
    treg = nisa.register_alloc()
    nisa.register_load(treg, t_sb)
    hb = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=hb, src=embed.ap(pattern=[[1, 128], [128, HT]], offset=0, scalar_offset=treg, indirect_dim=0))
    h_out = nl.ndarray((128, HT), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=h_out, src=hb)
    return h_out


BIGID = 1048576.0  # 2^20 > vocab: (BIGID - id) is exact in fp32


def head_body(h, pos, toks, gf, lm, idb, DEBUG=0, lgin=None):
    """final norm, LM head (vocab split over cores), argmax (ties -> lowest id), toks[pos+1] = argmax, returns pos+1."""
    p_sb, preg = _load_pos(pos)
    ones_f = nl.ndarray((128, 128), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    h_bf = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=h_bf, src=h)
    return head_sb(h_bf, p_sb, preg, ones_f, toks, gf, lm, idb, DEBUG, lgin)


def head_sb(h_bf, p_sb, preg, ones_f, toks, gf, lm, idb, DEBUG=0, lgin=None, ring=None, rb=0, NB=0, rw=0):
    """head_body on the SBUF hidden column; ring: stream the LM head through the decode layers' ring (fused NEFF)."""
    pid = nl.program_id(0)
    NV = VTC - HSK if pid == 0 else VTC + HSK   # this core's vocab tiles
    g_sb = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_sb, src=gf)
    xn = rmsnorm_col(h_bf, g_sb, ones_f)
    lg = nl.ndarray((128, NV), dtype=BF16, buffer=nl.sbuf)   # logits rounded to bf16 (as the reference)
    lm_c = lm[pid, :, :]
    W = HT * 128
    if ring is None and HEAD_RING_NB:
        ring = nl.ndarray((128, HEAD_RING_NB, HEAD_RW), dtype=BF16, buffer=nl.sbuf)
        NB = HEAD_RING_NB
        rw = HEAD_RW
    for n0 in range(0, NV, 128):
        n = min(128, NV - n0)
        ps = nl.ndarray((128, 128), dtype=F32, buffer=nl.psum)
        if ring is not None:
            rb = gemv_ring(lm_c[:, n0 * W:(n0 + n) * W], xn, n, HT, ps, ring, rb, NB, rw)
        else:
            gemv_cols(lm_c[:, n0 * W:(n0 + n) * W], xn, n, HT, ps)
        nisa.tensor_copy(dst=lg[:, n0:n0 + n], src=ps[:, 0:n])
    lgf = nl.ndarray((128, NV), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lgf, src=lg)
    if lgin != None:
        nisa.dma_copy(dst=lgf, src=lgin[pid, :, 0:NV])
    m8 = nl.ndarray((128, 8), dtype=F32, buffer=nl.sbuf)
    nisa.max8(dst=m8, src=lgf)
    i8 = nl.ndarray((128, 8), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=i8, data=lgf, vals=m8)
    cf = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cf, src=i8[:, 0:1])
    ib = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ib, src=idb[pid, :, :])
    idf = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)          # global id of the partition's first max
    nisa.scalar_tensor_tensor(dst=idf, data=cf, op0=nl.multiply, operand0=128.0, op1=nl.add, operand1=ib)
    # cross-partition: M = max, then lowest id among partitions reaching M
    mrow = nl.ndarray((1, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=mrow, op=nl.maximum, data=m8[:, 0:1])
    mrow_b = nl.ndarray((1, 1), dtype=BF16, buffer=nl.sbuf)      # exact: logits are bf16 values
    nisa.tensor_copy(dst=mrow_b, src=mrow)
    ones_b = nl.ndarray((1, 128), dtype=BF16, buffer=nl.sbuf)
    nisa.memset(dst=ones_b, value=1.0)
    psm = nl.ndarray((128, 1), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(dst=psm, stationary=ones_b, moving=mrow_b)
    mall = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mall, src=psm)
    eq = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=eq, data1=m8[:, 0:1], data2=mall, op=nl.equal)
    sc = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)           # (BIGID - id) where max, else 0
    nisa.tensor_scalar(dst=sc, data=idf, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=BIGID)
    nisa.tensor_tensor(dst=sc, data1=sc, data2=eq, op=nl.multiply)
    srow = nl.ndarray((1, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=srow, op=nl.maximum, data=sc)
    # cross-core combine of (M, score)
    loc = nl.ndarray((1, 2), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=loc[:, 0:1], src=mrow)
    nisa.tensor_copy(dst=loc[:, 1:2], src=srow)
    oth = nl.ndarray((1, 2), dtype=F32, buffer=nl.sbuf)
    nisa.sendrecv(src=loc, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    mg = nl.ndarray((1, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=mg, data1=loc[:, 0:1], data2=oth[:, 0:1], op=nl.maximum)
    e1 = nl.ndarray((1, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=e1, data1=loc[:, 0:1], data2=mg, op=nl.equal)
    nisa.tensor_tensor(dst=e1, data1=e1, data2=loc[:, 1:2], op=nl.multiply)
    e2 = nl.ndarray((1, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=e2, data1=oth[:, 0:1], data2=mg, op=nl.equal)
    nisa.tensor_tensor(dst=e2, data1=e2, data2=oth[:, 1:2], op=nl.multiply)
    best = nl.ndarray((1, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=best, data1=e1, data2=e2, op=nl.maximum)
    tokf = nl.ndarray((1, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tokf, data=best, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=BIGID)
    tok = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok, src=tokf)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=1, scalar_offset=preg, indirect_dim=1), src=tok)
    p1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p1, data=p_sb, op0=nl.add, operand0=1)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=pos_out, src=p1)
    if DEBUG:
        dbg = nl.ndarray((2, 128, VTCMAX), dtype=F32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=dbg[pid, :, 0:NV], src=lgf)
        tk = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=tk, src=tok)
        return pos_out, dbg, tk
    return pos_out


@nki.jit
def dec_head(h, pos, toks, gf, lm, idb, DEBUG=0):
    return head_body(h, pos, toks, gf, lm, idb, DEBUG)


@nki.jit
def dec_head_tietest(h, pos, toks, gf, lm, idb, lgin):
    return head_body(h, pos, toks, gf, lm, idb, 1, lgin)


def _grp_body(h, pos, cosT, sinT, consts, layers, B, emb=None, head=None):
    """16 decode layers; emb = (toks, embed): the input hidden is the embedding of toks[pos] (dec_embed fused);
    head = (toks, gf, lm, idb): the final norm, LM head and argmax follow (dec_head fused), returns pos+1."""
    p_sb, preg = _load_pos(pos)
    ones_f = nl.ndarray((128, 128), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    cst = nl.ndarray((128, 256), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=cst, src=consts)
    x = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    if emb is not None:
        t_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t_sb, src=emb[0].ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
        treg = nisa.register_alloc()
        nisa.register_load(treg, t_sb)
        nisa.dma_copy(dst=x, src=emb[1].ap(pattern=[[1, 128], [128, HT]], offset=0, scalar_offset=treg, indirect_dim=0))
    else:
        nisa.dma_copy(dst=x, src=h)
    nb, rw = _ring_cfg(B)
    ring = nl.ndarray((128, nb, rw), dtype=BF16, buffer=nl.sbuf)
    rb = 0
    xb = _xbuf()
    for lw in layers:
        x, rb = layer_sb(x, pos, preg, ones_f, cst, cosT, sinT, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6],
                         lw[7], B, 0, ring, rb, nb, rw, xb)
    if head is not None:
        return head_sb(x, p_sb, preg, ones_f, head[0], head[1], head[2], head[3], 0, None, ring, rb, nb, rw)
    h_out = nl.ndarray((128, HT), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=h_out, src=x)
    return h_out


# ---- generated: dec_grp<G> kernels (NKI parses the source, so the signatures are spelled out)
@nki.jit
def dec_grp16(h, pos, cosT, sinT, consts,
              g1_0, wqkv_0, wo_0, g2_0, wgu_0, wd_0, kc_0, vc_0,
              g1_1, wqkv_1, wo_1, g2_1, wgu_1, wd_1, kc_1, vc_1,
              g1_2, wqkv_2, wo_2, g2_2, wgu_2, wd_2, kc_2, vc_2,
              g1_3, wqkv_3, wo_3, g2_3, wgu_3, wd_3, kc_3, vc_3,
              g1_4, wqkv_4, wo_4, g2_4, wgu_4, wd_4, kc_4, vc_4,
              g1_5, wqkv_5, wo_5, g2_5, wgu_5, wd_5, kc_5, vc_5,
              g1_6, wqkv_6, wo_6, g2_6, wgu_6, wd_6, kc_6, vc_6,
              g1_7, wqkv_7, wo_7, g2_7, wgu_7, wd_7, kc_7, vc_7,
              g1_8, wqkv_8, wo_8, g2_8, wgu_8, wd_8, kc_8, vc_8,
              g1_9, wqkv_9, wo_9, g2_9, wgu_9, wd_9, kc_9, vc_9,
              g1_10, wqkv_10, wo_10, g2_10, wgu_10, wd_10, kc_10, vc_10,
              g1_11, wqkv_11, wo_11, g2_11, wgu_11, wd_11, kc_11, vc_11,
              g1_12, wqkv_12, wo_12, g2_12, wgu_12, wd_12, kc_12, vc_12,
              g1_13, wqkv_13, wo_13, g2_13, wgu_13, wd_13, kc_13, vc_13,
              g1_14, wqkv_14, wo_14, g2_14, wgu_14, wd_14, kc_14, vc_14,
              g1_15, wqkv_15, wo_15, g2_15, wgu_15, wd_15, kc_15, vc_15,
              B=512):
    """16 consecutive decode layers in one NEFF."""
    return _grp_body(h, pos, cosT, sinT, consts, [
        (g1_0, wqkv_0, wo_0, g2_0, wgu_0, wd_0, kc_0, vc_0),
        (g1_1, wqkv_1, wo_1, g2_1, wgu_1, wd_1, kc_1, vc_1),
        (g1_2, wqkv_2, wo_2, g2_2, wgu_2, wd_2, kc_2, vc_2),
        (g1_3, wqkv_3, wo_3, g2_3, wgu_3, wd_3, kc_3, vc_3),
        (g1_4, wqkv_4, wo_4, g2_4, wgu_4, wd_4, kc_4, vc_4),
        (g1_5, wqkv_5, wo_5, g2_5, wgu_5, wd_5, kc_5, vc_5),
        (g1_6, wqkv_6, wo_6, g2_6, wgu_6, wd_6, kc_6, vc_6),
        (g1_7, wqkv_7, wo_7, g2_7, wgu_7, wd_7, kc_7, vc_7),
        (g1_8, wqkv_8, wo_8, g2_8, wgu_8, wd_8, kc_8, vc_8),
        (g1_9, wqkv_9, wo_9, g2_9, wgu_9, wd_9, kc_9, vc_9),
        (g1_10, wqkv_10, wo_10, g2_10, wgu_10, wd_10, kc_10, vc_10),
        (g1_11, wqkv_11, wo_11, g2_11, wgu_11, wd_11, kc_11, vc_11),
        (g1_12, wqkv_12, wo_12, g2_12, wgu_12, wd_12, kc_12, vc_12),
        (g1_13, wqkv_13, wo_13, g2_13, wgu_13, wd_13, kc_13, vc_13),
        (g1_14, wqkv_14, wo_14, g2_14, wgu_14, wd_14, kc_14, vc_14),
        (g1_15, wqkv_15, wo_15, g2_15, wgu_15, wd_15, kc_15, vc_15),
    ], B)


@nki.jit
def dec_grp16e(toks, pos, embed, cosT, sinT, consts,
              g1_0, wqkv_0, wo_0, g2_0, wgu_0, wd_0, kc_0, vc_0,
              g1_1, wqkv_1, wo_1, g2_1, wgu_1, wd_1, kc_1, vc_1,
              g1_2, wqkv_2, wo_2, g2_2, wgu_2, wd_2, kc_2, vc_2,
              g1_3, wqkv_3, wo_3, g2_3, wgu_3, wd_3, kc_3, vc_3,
              g1_4, wqkv_4, wo_4, g2_4, wgu_4, wd_4, kc_4, vc_4,
              g1_5, wqkv_5, wo_5, g2_5, wgu_5, wd_5, kc_5, vc_5,
              g1_6, wqkv_6, wo_6, g2_6, wgu_6, wd_6, kc_6, vc_6,
              g1_7, wqkv_7, wo_7, g2_7, wgu_7, wd_7, kc_7, vc_7,
              g1_8, wqkv_8, wo_8, g2_8, wgu_8, wd_8, kc_8, vc_8,
              g1_9, wqkv_9, wo_9, g2_9, wgu_9, wd_9, kc_9, vc_9,
              g1_10, wqkv_10, wo_10, g2_10, wgu_10, wd_10, kc_10, vc_10,
              g1_11, wqkv_11, wo_11, g2_11, wgu_11, wd_11, kc_11, vc_11,
              g1_12, wqkv_12, wo_12, g2_12, wgu_12, wd_12, kc_12, vc_12,
              g1_13, wqkv_13, wo_13, g2_13, wgu_13, wd_13, kc_13, vc_13,
              g1_14, wqkv_14, wo_14, g2_14, wgu_14, wd_14, kc_14, vc_14,
              g1_15, wqkv_15, wo_15, g2_15, wgu_15, wd_15, kc_15, vc_15,
              B=512):
    """dec_embed + 16 decode layers in one NEFF."""
    return _grp_body(None, pos, cosT, sinT, consts, [
        (g1_0, wqkv_0, wo_0, g2_0, wgu_0, wd_0, kc_0, vc_0),
        (g1_1, wqkv_1, wo_1, g2_1, wgu_1, wd_1, kc_1, vc_1),
        (g1_2, wqkv_2, wo_2, g2_2, wgu_2, wd_2, kc_2, vc_2),
        (g1_3, wqkv_3, wo_3, g2_3, wgu_3, wd_3, kc_3, vc_3),
        (g1_4, wqkv_4, wo_4, g2_4, wgu_4, wd_4, kc_4, vc_4),
        (g1_5, wqkv_5, wo_5, g2_5, wgu_5, wd_5, kc_5, vc_5),
        (g1_6, wqkv_6, wo_6, g2_6, wgu_6, wd_6, kc_6, vc_6),
        (g1_7, wqkv_7, wo_7, g2_7, wgu_7, wd_7, kc_7, vc_7),
        (g1_8, wqkv_8, wo_8, g2_8, wgu_8, wd_8, kc_8, vc_8),
        (g1_9, wqkv_9, wo_9, g2_9, wgu_9, wd_9, kc_9, vc_9),
        (g1_10, wqkv_10, wo_10, g2_10, wgu_10, wd_10, kc_10, vc_10),
        (g1_11, wqkv_11, wo_11, g2_11, wgu_11, wd_11, kc_11, vc_11),
        (g1_12, wqkv_12, wo_12, g2_12, wgu_12, wd_12, kc_12, vc_12),
        (g1_13, wqkv_13, wo_13, g2_13, wgu_13, wd_13, kc_13, vc_13),
        (g1_14, wqkv_14, wo_14, g2_14, wgu_14, wd_14, kc_14, vc_14),
        (g1_15, wqkv_15, wo_15, g2_15, wgu_15, wd_15, kc_15, vc_15),
    ], B, emb=(toks, embed))


@nki.jit
def dec_grp16h(h, pos, cosT, sinT, consts,
              g1_0, wqkv_0, wo_0, g2_0, wgu_0, wd_0, kc_0, vc_0,
              g1_1, wqkv_1, wo_1, g2_1, wgu_1, wd_1, kc_1, vc_1,
              g1_2, wqkv_2, wo_2, g2_2, wgu_2, wd_2, kc_2, vc_2,
              g1_3, wqkv_3, wo_3, g2_3, wgu_3, wd_3, kc_3, vc_3,
              g1_4, wqkv_4, wo_4, g2_4, wgu_4, wd_4, kc_4, vc_4,
              g1_5, wqkv_5, wo_5, g2_5, wgu_5, wd_5, kc_5, vc_5,
              g1_6, wqkv_6, wo_6, g2_6, wgu_6, wd_6, kc_6, vc_6,
              g1_7, wqkv_7, wo_7, g2_7, wgu_7, wd_7, kc_7, vc_7,
              g1_8, wqkv_8, wo_8, g2_8, wgu_8, wd_8, kc_8, vc_8,
              g1_9, wqkv_9, wo_9, g2_9, wgu_9, wd_9, kc_9, vc_9,
              g1_10, wqkv_10, wo_10, g2_10, wgu_10, wd_10, kc_10, vc_10,
              g1_11, wqkv_11, wo_11, g2_11, wgu_11, wd_11, kc_11, vc_11,
              g1_12, wqkv_12, wo_12, g2_12, wgu_12, wd_12, kc_12, vc_12,
              g1_13, wqkv_13, wo_13, g2_13, wgu_13, wd_13, kc_13, vc_13,
              g1_14, wqkv_14, wo_14, g2_14, wgu_14, wd_14, kc_14, vc_14,
              g1_15, wqkv_15, wo_15, g2_15, wgu_15, wd_15, kc_15, vc_15,
              toks, gf, lm, idb, B=512):
    """16 decode layers + dec_head in one NEFF."""
    return _grp_body(h, pos, cosT, sinT, consts, [
        (g1_0, wqkv_0, wo_0, g2_0, wgu_0, wd_0, kc_0, vc_0),
        (g1_1, wqkv_1, wo_1, g2_1, wgu_1, wd_1, kc_1, vc_1),
        (g1_2, wqkv_2, wo_2, g2_2, wgu_2, wd_2, kc_2, vc_2),
        (g1_3, wqkv_3, wo_3, g2_3, wgu_3, wd_3, kc_3, vc_3),
        (g1_4, wqkv_4, wo_4, g2_4, wgu_4, wd_4, kc_4, vc_4),
        (g1_5, wqkv_5, wo_5, g2_5, wgu_5, wd_5, kc_5, vc_5),
        (g1_6, wqkv_6, wo_6, g2_6, wgu_6, wd_6, kc_6, vc_6),
        (g1_7, wqkv_7, wo_7, g2_7, wgu_7, wd_7, kc_7, vc_7),
        (g1_8, wqkv_8, wo_8, g2_8, wgu_8, wd_8, kc_8, vc_8),
        (g1_9, wqkv_9, wo_9, g2_9, wgu_9, wd_9, kc_9, vc_9),
        (g1_10, wqkv_10, wo_10, g2_10, wgu_10, wd_10, kc_10, vc_10),
        (g1_11, wqkv_11, wo_11, g2_11, wgu_11, wd_11, kc_11, vc_11),
        (g1_12, wqkv_12, wo_12, g2_12, wgu_12, wd_12, kc_12, vc_12),
        (g1_13, wqkv_13, wo_13, g2_13, wgu_13, wd_13, kc_13, vc_13),
        (g1_14, wqkv_14, wo_14, g2_14, wgu_14, wd_14, kc_14, vc_14),
        (g1_15, wqkv_15, wo_15, g2_15, wgu_15, wd_15, kc_15, vc_15),
    ], B, head=(toks, gf, lm, idb))
