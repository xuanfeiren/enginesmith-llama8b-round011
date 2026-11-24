"""Prefill NEFFs for a prompt bucket of S tokens (positions 0..S-1), layered: embedding, one layer (run 32x on
per-layer slices), pick (last token's hidden -> decode head).

Activations are feature-major in HBM: x [128 p, 32 kt, S] = hidden feature kt*128+p of token s.
The two physical cores split each layer like the decode step (heads / intermediate tiles), see pf_layer_body.
"""
import nki
import nki.language as nl
import nki.isa as nisa

from common import (H, HT, HD, IT, ITC, TMAX, QKV_NT, GU_G, BF16, F32, EPS, wdma, gemv_cols, rmsnorm_col)
import decode
from nkattn.attention.attention_cte import _attention_cte   # nkilib (Apache-2.0), see THIRD_PARTY.md
import pattn

QSCALE = 1.0 / (HD ** 0.5)
PF_TOK_SYNC_X = 1   # S >= 1024: phase-A x chunks on the Sync queue (ahead of the weights) + split norm
PF_TOK_SQ_STREAM = 1   # S >= 2048 phase C: MLP-norm squares summed per o_proj output tile (bitwise the same)
PF_PIPE_SQ_STREAM = 1  # S = 1024 (pf_layer_pipe): the same
PF_ATTN_T_MIN = 2048   # S >= this: transposed attention (pattn.attention_t, round 009), else attention_heads
PF_ATTN = 1     # 1: pattn.attention_heads (same bytes as nkilib, per-group stages); 0: nkilib attention_cte
NS_SHIFT = 0    # S <= 256: gate/up tiles moved from core 0 (the slower one) to core 1
NS_OWNF = 1     # S <= 256: own attention heads kept in SBUF; o_proj / down accumulate the own K half first, 4 output tiles at a time
NS_NB = 8
NS_POSTR = 1   # S <= 256: GEMMs after a norm on bf16(x * g), rstd applied to the fp32 result (not bitwise)
NS_NBG = 2   # gate/up tiles per own-K-first batch (NS_PSX)
NS_SQ_EARLY = 1   # S <= 256: own half of h1 squared before the h1 exchange
NS_HALF = 0       # S = 256: norm scaling by token halves; the first NS_HALF GEMM tiles after it in halves
NS_DVSQ = 1       # S <= 256: half of the norm's squares on the Vector engine
NS_NGPS = 0       # S <= 256: hidden tiles of the norm scaling on GpSimd
PF_E_RING = 8   # S <= 256 down projection: weight slabs through an explicit ring of this many slots (0: off)


def _attention(qs, kq, vq, att):
    """Causal attention as the reference (NxD: nkilib attention_cte, q pre-scaled by 1/sqrt(d), scale 1).
    qs [32, 128, S] (pre-scaled q^T), kq [8, 128, S], vq [8, S, 128] -> att [32, 128, S] (o^T per head).
    Sharded over the two cores by heads (core c: q heads 16c.., kv heads 4c..)."""
    if PF_ATTN and qs.shape[2] >= PF_ATTN_T_MIN:
        pattn.attention_t(qs, kq, vq, att)
        return
    if PF_ATTN:
        pattn.attention_heads(qs, kq, vq, att)
        return
    _attention_cte(q=qs, k=kq, v=vq, scale=1.0, causal_mask=True, tp_q=False, tp_k=False, tp_out=True,
                   out_o_hbm=att)


def adma(dst, src):
    """Activation exchange / load on the Scalar engine's hardware DGE queue (the Sync queue streams weights;
    GpSimd software DGE was the default)."""
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)


NS_XQM = 0  # queue of the MLP-intermediate exchange load (same codes)
NS_XQH = 0  # queue of the h1 exchange load
NS_XQ = 0   # S <= 256: exchange loads after a core barrier on the GpSimd queue (not ahead of the weights on Sync)


NS_SQ = 0   # queue of the per-kv-group attention stores: 0 GpSimd (default DMA), 1 Sync, 2 Scalar


def sdma(dst, src):
    if NS_SQ == 1:
        wdma(dst, src)
    elif NS_SQ == 2:
        adma(dst, src)
    else:
        nisa.dma_copy(dst=dst, src=src)


NS_STQ = 2    # S <= 256: queue of the h1 / MLP-intermediate exchange stores: 0 GpSimd, 1 Sync, 2 Scalar
NS_STQX = 2   # queue of the partial-sum store (NS_PSX)


def stq(dst, src, q=None):
    if q is None:
        q = NS_STQ
    if q == 1:
        wdma(dst, src)
    elif q == 2:
        adma(dst, src)
    else:
        nisa.dma_copy(dst=dst, src=src)


def xdma(dst, src, q=None):
    if q is None:
        q = NS_XQ
    if q == 2:
        adma(dst, src)
    elif q:
        nisa.dma_copy(dst=dst, src=src)
    else:
        wdma(dst, src)


def _consts(consts):
    cst = nl.ndarray((128, 256), dtype=BF16, buffer=nl.sbuf)   # [rperm | identity]
    nisa.dma_copy(dst=cst, src=consts)
    return cst


def _norm_fm(x, g_f, ones_b, C):
    """RMSNorm of a feature-major chunk x [128, HT, C] bf16 -> bf16(x * g * rstd)."""
    sq = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=x)
    ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=ps, stationary=ones_b, moving=sq[:, kt, :])
    t = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=ps, op0=nl.multiply, operand0=1.0 / H, op1=nl.add, operand1=EPS)
    rstd = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=t, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    xn = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
    for kt in range(HT):
        nisa.scalar_tensor_tensor(dst=xn[:, kt, :], data=x[:, kt, :], op0=nl.multiply, operand0=g_f[:, kt:kt + 1],
                                  op1=nl.multiply, operand1=rstd)
    return xn


def _norm_fm_from_sum(x, g_f, ps, C):
    """_norm_fm with the column sums of the squares already in ps (PSUM [128, C], every partition the same)."""
    t = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=ps, op0=nl.multiply, operand0=1.0 / H, op1=nl.add, operand1=EPS)
    rstd = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=t, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    xn = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
    for kt in range(HT):
        nisa.scalar_tensor_tensor(dst=xn[:, kt, :], data=x[:, kt, :], op0=nl.multiply, operand0=g_f[:, kt:kt + 1],
                                  op1=nl.multiply, operand1=rstd)
    return xn


def _gemm_tile(ps, wsl, xin, KT):
    """ps [128 n, C] = sum_kt wsl tile kt ([128 k, 128 n]) ^T @ xin[:, kt, :]."""
    for kt in range(KT):
        nisa.nc_matmul(dst=ps, stationary=wsl[:, kt * 128:(kt + 1) * 128], moving=xin[:, kt, :])


def _gemm_tile_halves(ps, wsl, xin, KT, C):
    """_gemm_tile as two half-width passes (token half 0 first): each column's sum is the same."""
    for hh in range(2):
        c0 = hh * (C // 2)
        for kt in range(KT):
            nisa.nc_matmul(dst=ps[:, c0:c0 + C // 2], stationary=wsl[:, kt * 128:(kt + 1) * 128],
                           moving=xin[:, kt, c0:c0 + C // 2])


def _load_slab(src2d, col0, ntiles):
    t = nl.ndarray((128, ntiles * 128), dtype=BF16, buffer=nl.sbuf)
    wdma(t, src2d[:, col0 * 128:(col0 + ntiles) * 128])
    return t


def pf_layer_body(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, xa, xb, hs, S, CD,
                  NS=0, ms=None, pos=None, xsb=None, xdst=None, xown=None):
    """Prefill layer over S tokens, split over the two cores like the decode step: QKV by heads (q 16c.., kv 4c..),
    attention for the core's own heads (all queries), o_proj and down K-split with fp32 partials exchanged through
    HBM scratch (xa, xb [2, 128, HT, S]) and a core barrier, gate/up N-split over intermediate tiles 56c..
    Every weight tile is read by one core only (once per token chunk)."""
    pid = nl.program_id(0)
    CA = min(512, S)           # token chunk, phase A (QKV)
    CC = min(512, S)           # phase C (o_proj partial)
    QB = min(128, S)           # query block
    scale = 1.0 / (HD ** 0.5)

    cst = _consts(consts)
    ones_b = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
    nisa.memset(dst=ones_b, value=1.0)
    g1b = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=g1b, src=g1)
    g1f = nl.ndarray((128, HT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g1f, src=g1b)
    g2b = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=g2b, src=g2)
    g2f = nl.ndarray((128, HT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g2f, src=g2b)
    wq_c = wqkv[pid, :, :]
    wo_c = wo[pid, :, :]
    wgu_c = wgu[pid, :, :]
    wd_c = wd[pid, :, :]

    if NS and NS_PIPE:   # phases A + B pipelined (S <= 256)
        acb = None
        if NS_OWNF:
            acb = nl.ndarray((128, HT, S), dtype=BF16, buffer=nl.sbuf)
        att2 = None
        if NS_ATT2 and NS_OWNF and pos is None:
            att2 = qs   # unused by the S <= 256 layers otherwise
        _ns_phase_ab(x, cosP, sinP, cst, g1f, ones_b, wq_c, kc, vc, att, S, acb, xsb, xown, att2)
        if pos is not None:
            return _pf_tail(x, wo, g2f, wgu, wd, att, hs, ms, ones_b, S, pos, barrier=True)
        return _pf_cde_nsplit(x, wo, g2f, wgu, wd, att, hs, ms, ones_b, S, acb, xa, xsb, xdst, xown, att2)

    # ================= phase A: norm, own QKV heads, RoPE, K/V cache writes, Q scratch
    for ch in range(S // CA):
        C = CA
        t0 = ch * C
        xc = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        if NS:   # x in 4 pieces on the Scalar queue; the norm's squaring pipelines behind them
            for a in range(4):
                wdma(xc[:, a * 8:(a + 1) * 8, :], x[:, a * 8:(a + 1) * 8, t0:t0 + C])
            xn = _norm_fm_split(xc, g1f, ones_b, C)
        else:
            nisa.dma_copy(dst=xc, src=x[:, :, t0:t0 + C])
            xn = _norm_fm(xc, g1f, ones_b, C)
        cs = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cs, src=cosP[:, t0:t0 + C])
        sn = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sn, src=sinP[:, t0:t0 + C])
        for n in range(QKV_NT):
            wsl = _load_slab(wq_c, n * HT, HT)
            ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            _gemm_tile(ps, wsl, xn, HT)
            yb = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=yb, src=ps)
            if n < 20:
                pr = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
                nisa.nc_matmul(dst=pr, stationary=cst[:, 0:128], moving=yb)
                a_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a_, data1=yb, data2=cs, op=nl.multiply)
                b_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=b_, data1=pr, data2=sn, op=nl.multiply)
                r_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=r_, data1=a_, data2=b_, op=nl.add)
                if n < 16:
                    q_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)   # q / sqrt(d) in bf16, as NxD
                    nisa.tensor_scalar(dst=q_, data=r_, op0=nl.multiply, operand0=QSCALE)
                    nisa.dma_copy(dst=qs[16 * pid + n, :, t0:t0 + C], src=q_)
                else:
                    nisa.dma_copy(dst=kc[4 * pid + n - 16, :, t0:t0 + C], src=r_)
                    nisa.dma_copy(dst=kq[4 * pid + n - 16, :, t0:t0 + C], src=r_)
            else:
                G = 4 * pid + n - 20
                for b0 in range(0, C, 128):
                    w = min(128, C - b0)
                    pt = nl.ndarray((128, 128), dtype=F32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pt[0:w, :], stationary=yb[:, b0:b0 + w], moving=cst[:, 128:256])
                    vt = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=vt[0:w, :], src=pt[0:w, :])
                    nisa.dma_copy(dst=vc[t0 + b0:t0 + b0 + w, G * 128:(G + 1) * 128], src=vt[0:w, :])
                    nisa.dma_copy(dst=vq[G, t0 + b0:t0 + b0 + w, :], src=vt[0:w, :])

    # ================= phase B: causal attention for the core's own heads (its own K/V/Q: no barrier needed)
    _attention(qs, kq, vq, att)
    if pos is not None:
        return _pf_tail(x, wo, g2f, wgu, wd, att, hs, ms, ones_b, S, pos, barrier=True)
    if NS:
        return _pf_cde_nsplit(x, wo, g2f, wgu, wd, att, hs, ms, ones_b, S)

    # ================= phase C: o_proj partial over the core's 16 heads -> xa[pid]
    for ch in range(S // CC):
        C = CC
        t0 = ch * C
        ac = nl.ndarray((128, 16, C), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=ac, src=att.ap(pattern=[[S, 128], [128 * S, 16], [1, C]], offset=16 * pid * 128 * S + t0))
        for nt in range(HT):
            wsl = _load_slab(wo_c, nt * 16, 16)
            ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            _gemm_tile(ps, wsl, ac, 16)
            pf = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pf, src=ps)
            nisa.dma_copy(dst=xa[pid, :, nt, t0:t0 + C], src=pf)
    nisa.core_barrier(data=xa, cores=(0, 1))

    # ================= phase D: h1 = x + o; norm; own gate/up; SiLU*up; down partial -> xb[pid]
    for ch in range(S // CD):
        C = CD
        t0 = ch * C
        h1 = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        xc = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xc, src=x[:, :, t0:t0 + C])
        for nt in range(HT):
            p0 = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
            nisa.dma_copy(dst=p0, src=xa[0, :, nt, t0:t0 + C])
            p1 = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
            nisa.dma_copy(dst=p1, src=xa[1, :, nt, t0:t0 + C])
            ob = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=ob, data1=p0, data2=p1, op=nl.add)
            nisa.tensor_tensor(dst=h1[:, nt, :], data1=xc[:, nt, :], data2=ob, op=nl.add)
        nisa.dma_copy(dst=hs[:, 16 * pid:16 * pid + 16, t0:t0 + C], src=h1[:, 16 * pid:16 * pid + 16, :])
        xn2 = _norm_fm(h1, g2f, ones_b, C)
        mT = nl.ndarray((128, ITC, C), dtype=BF16, buffer=nl.sbuf)
        for gi in range(ITC // GU_G):
            for j in range(GU_G):
                wg = _load_slab(wgu_c, (gi * 2 * GU_G + j) * HT, HT)
                wu = _load_slab(wgu_c, (gi * 2 * GU_G + GU_G + j) * HT, HT)
                pg = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
                _gemm_tile(pg, wg, xn2, HT)
                pu = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
                _gemm_tile(pu, wu, xn2, HT)
                gb = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gb, src=pg)
                ub = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ub, src=pu)
                ab = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
                nisa.activation(dst=ab, op=nl.silu, data=gb)
                nisa.tensor_tensor(dst=mT[:, gi * GU_G + j, :], data1=ab, data2=ub, op=nl.multiply)
        for nt in range(HT):
            wsl = _load_slab(wd_c, nt * ITC, ITC)
            ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            _gemm_tile(ps, wsl, mT, ITC)
            pf = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pf, src=ps)
            nisa.dma_copy(dst=xb[pid, :, nt, t0:t0 + C], src=pf)
    nisa.core_barrier(data=xb, cores=(0, 1))

    # ================= phase E: x_out = h1 + down (this core's half of the features)
    x_out = nl.ndarray((128, HT, S), dtype=BF16, buffer=nl.shared_hbm)
    CE = min(512, S)
    for ch in range(S // CE):
        C = CE
        t0 = ch * C
        hc = nl.ndarray((128, 16, C), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=hc, src=hs[:, 16 * pid:16 * pid + 16, t0:t0 + C])
        oc = nl.ndarray((128, 16, C), dtype=BF16, buffer=nl.sbuf)
        for i in range(16):
            nt = 16 * pid + i
            p0 = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
            nisa.dma_copy(dst=p0, src=xb[0, :, nt, t0:t0 + C])
            p1 = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
            nisa.dma_copy(dst=p1, src=xb[1, :, nt, t0:t0 + C])
            db = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=db, data1=p0, data2=p1, op=nl.add)
            nisa.tensor_tensor(dst=oc[:, i, :], data1=hc[:, i, :], data2=db, op=nl.add)
        nisa.dma_copy(dst=x_out[:, 16 * pid:16 * pid + 16, t0:t0 + C], src=oc)
    return x_out


def _norm_fm_split(x, g_f, ones_b, C, NSQ=4, ngps=0, sq=None, sq_done=None, halves=False, dvsq=False):
    """_norm_fm with the squaring split into NSQ instructions (elementwise: same bytes), so the chain
    load -> square -> sum -> rstd -> scale -> first GEMM pipelines over hidden-tile groups.
    ngps: the scaling of every (32 // ngps)-th hidden tile runs on GpSimd instead of the Vector engine (the same
    two fp32 multiplies and bf16 rounding). sq / sq_done: squares already computed for those NSQ pieces."""
    if sq is None:
        sq = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
    st = HT // NSQ
    for a in range(NSQ):
        if sq_done is None or not sq_done[a]:
            if dvsq and a % 2 == 1:   # x * x rounded to bf16 on the Vector engine (x^2 is exact in fp32: same bytes)
                nisa.tensor_tensor(dst=sq[:, a * st:(a + 1) * st, :], data1=x[:, a * st:(a + 1) * st, :],
                                   data2=x[:, a * st:(a + 1) * st, :], op=nl.multiply)
            else:
                nisa.activation(dst=sq[:, a * st:(a + 1) * st, :], op=nl.square, data=x[:, a * st:(a + 1) * st, :])
    ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=ps, stationary=ones_b, moving=sq[:, kt, :])
    t = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=ps, op0=nl.multiply, operand0=1.0 / H, op1=nl.add, operand1=EPS)
    rstd = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=t, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    xn = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
    if halves:   # token half 0 of every hidden tile first (columns are independent: same bytes)
        for hh in range(2):
            c0 = hh * (C // 2)
            for kt in range(HT):
                nisa.scalar_tensor_tensor(dst=xn[:, kt, c0:c0 + C // 2], data=x[:, kt, c0:c0 + C // 2], op0=nl.multiply,
                                          operand0=g_f[:, kt:kt + 1], op1=nl.multiply, operand1=rstd[:, c0:c0 + C // 2])
        return xn
    for kt in range(HT):
        if ngps and kt % (HT // ngps) == HT // ngps - 1:
            xg = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=xg, data=x[:, kt, :], op0=nl.multiply, operand0=g_f[:, kt:kt + 1],
                               engine=nisa.engine.gpsimd)
            nisa.tensor_tensor(dst=xn[:, kt, :], data1=xg, data2=rstd, op=nl.multiply, engine=nisa.engine.gpsimd)
        else:
            nisa.scalar_tensor_tensor(dst=xn[:, kt, :], data=x[:, kt, :], op0=nl.multiply, operand0=g_f[:, kt:kt + 1],
                                      op1=nl.multiply, operand1=rstd)
    return xn


def _rstd_fm(x, ones_b, C, dvsq=False):
    """rstd [128, C] fp32 of a feature-major [128, HT, C] bf16 block (the statistics of _norm_fm_split)."""
    sq = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
    st = HT // 4
    for a in range(4):
        if dvsq and a % 2 == 1:
            nisa.tensor_tensor(dst=sq[:, a * st:(a + 1) * st, :], data1=x[:, a * st:(a + 1) * st, :],
                               data2=x[:, a * st:(a + 1) * st, :], op=nl.multiply)
        else:
            nisa.activation(dst=sq[:, a * st:(a + 1) * st, :], op=nl.square, data=x[:, a * st:(a + 1) * st, :])
    ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=ps, stationary=ones_b, moving=sq[:, kt, :])
    t = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=ps, op0=nl.multiply, operand0=1.0 / H, op1=nl.add, operand1=EPS)
    rstd = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=t, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    return rstd


def _scale_g(dst, x, g_f, kt0, kt1):
    """dst[:, kt] = bf16(x[:, kt] * g[kt]) for kt0 <= kt < kt1 (alternating Vector / Scalar engines)."""
    for kt in range(kt0, kt1):
        if kt % 2 == 0:
            nisa.tensor_scalar(dst=dst[:, kt, :], data=x[:, kt, :], op0=nl.multiply, operand0=g_f[:, kt:kt + 1])
        else:
            nisa.activation(dst=dst[:, kt, :], op=nl.copy, data=x[:, kt, :], scale=g_f[:, kt:kt + 1])


def _pf_cde_nsplit(x, wo, g2f, wgu, wd, att, hs, ms, ones_b, S, acb=None, xa=None, xsb=None, xdst=None, xown=None,
                   att2=None):
    """Phases C..E with o_proj and down split along N (core c: hidden tiles 16c..16c+15), S <= 256 tokens in one
    chunk. Exchanges bf16 activations instead of fp32 partials: attention output (att), h1 halves (hs), the MLP
    intermediate (ms [128, IT, S]); weights are read from both per-core runs of the decode (K-split) layout.
    Round 003: each core keeps its own halves of h1 and of the MLP intermediate in SBUF and reads only the other
    core's half back; the exchanges and the output are stored in pieces as they are produced (same arithmetic)."""
    pid = nl.program_id(0)
    o = 1 - pid
    C = S
    nisa.core_barrier(data=att, cores=(0, 1))
    # ---- C: h1 (own half) = x + o_proj(att)
    if acb is not None and att2 is not None:   # the other core's first 12 heads, then its last 4 (own barrier)
        ac = acb
        xdma(dst=ac[:, 16 * o:16 * o + 12, :], src=att.ap(pattern=[[S, 128], [128 * S, 12], [1, C]],
                                                          offset=16 * o * 128 * S))
        nisa.core_barrier(data=att2, cores=(0, 1))
        xdma(dst=ac[:, 16 * o + 12:16 * o + 16, :], src=att2.ap(pattern=[[S, 128], [128 * S, 4], [1, C]],
                                                                offset=(16 * o + 12) * 128 * S))
    elif acb is not None:   # own heads already in SBUF: load the other core's 16 heads only
        ac = acb
        xdma(dst=ac[:, 16 * o:16 * o + 16, :], src=att.ap(pattern=[[S, 128], [128 * S, 16], [1, C]],
                                                          offset=16 * o * 128 * S))
    else:
        ac = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        wdma(dst=ac, src=att.ap(pattern=[[S, 128], [128 * S, HT], [1, C]], offset=0))
    if xsb is not None:   # x in SBUF (pf_first_ns): own feature half at offset 16 pid
        xc = xsb
        xo = 16 * pid
    elif xown is not None:   # own half of x in SBUF (previous layer of a pf_multi NEFF)
        xc = xown
        xo = 0
    else:
        xc = nl.ndarray((128, 16, C), dtype=BF16, buffer=nl.sbuf)
        adma(dst=xc, src=x[:, 16 * pid:16 * pid + 16, 0:C])
        xo = 0
    h1 = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
    xn2x = None
    if NS_XN2X and NS_POSTR and NS_PSX and xa is not None and NS_OWNF:
        xn2x = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
    if NS_OWNF:
        ccs = [pid, o]
        for b in range(16 // NS_NB):
            pss = []
            for i in range(NS_NB):
                pss.append(nl.ndarray((128, C), dtype=F32, buffer=nl.psum))
            for cc in ccs:
                for i in range(NS_NB):
                    nt = 16 * pid + b * NS_NB + i
                    wsl = _load_slab(wo[cc, :, :], nt * 16, 16)
                    for k in range(16):
                        nisa.nc_matmul(dst=pss[i], stationary=wsl[:, k * 128:(k + 1) * 128],
                                       moving=ac[:, cc * 16 + k, :])
            for i in range(NS_NB):
                nt = 16 * pid + b * NS_NB + i
                ob = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ob, src=pss[i])
                nisa.tensor_tensor(dst=h1[:, nt, :], data1=xc[:, xo + b * NS_NB + i, :], data2=ob, op=nl.add)
            nt0 = 16 * pid + b * NS_NB
            if xn2x is not None:   # exchange bf16(h1 * g2) (what the other core's gate/up reads) instead of h1
                _scale_g(xn2x, h1, g2f, nt0, nt0 + NS_NB)
                stq(dst=hs[:, nt0:nt0 + NS_NB, 0:C], src=xn2x[:, nt0:nt0 + NS_NB, :])
            else:
                stq(dst=hs[:, nt0:nt0 + NS_NB, 0:C], src=h1[:, nt0:nt0 + NS_NB, :])
    for i in range(0 if NS_OWNF else 16):
        nt = 16 * pid + i
        ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
        for cc in range(2):
            wsl = _load_slab(wo[cc, :, :], nt * 16, 16)
            for k in range(16):
                nisa.nc_matmul(dst=ps, stationary=wsl[:, k * 128:(k + 1) * 128], moving=ac[:, cc * 16 + k, :])
        ob = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ob, src=ps)
        nisa.tensor_tensor(dst=h1[:, nt, :], data1=xc[:, xo + i, :], data2=ob, op=nl.add)
        if i % 4 == 3:
            adma(dst=hs[:, nt - 3:nt + 1, 0:C], src=h1[:, nt - 3:nt + 1, :])
    ma = nl.ndarray((128, IT, C), dtype=BF16, buffer=nl.sbuf)
    lo = 0 if pid == 0 else ITC - NS_SHIFT
    hi = ITC - NS_SHIFT if pid == 0 else IT
    if NS_PSX and xa is not None:
        _ns_d_psx(h1, g2f, ones_b, wgu, hs, ms, xa, ma, lo, hi, S, xn2x)
    else:
        sq2 = None
        sqd = None
        if NS_SQ_EARLY:   # squares of the own half before the exchange (elementwise: same bytes)
            sq2 = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
            sqd = [False, False, False, False]
            for a in range(2):
                a2 = 2 * pid + a
                nisa.activation(dst=sq2[:, a2 * 8:(a2 + 1) * 8, :], op=nl.square, data=h1[:, a2 * 8:(a2 + 1) * 8, :])
                sqd[a2] = True
        nisa.core_barrier(data=hs, cores=(0, 1))
        xdma(dst=h1[:, 16 * o:16 * o + 16, :], src=hs[:, 16 * o:16 * o + 16, 0:C], q=NS_XQH)
        # ---- D: norm(h1), this core's gate/up tiles (core 0: 0 .. ITC-NS_SHIFT-1, core 1: the rest), SiLU * up -> ma, ms
        xn2 = _norm_fm_split(h1, g2f, ones_b, C, ngps=NS_NGPS, sq=sq2, sq_done=sqd, halves=NS_HALF > 0, dvsq=NS_DVSQ)
        m0 = lo
        for it in range(lo, hi):
            cc, il = it // ITC, it % ITC
            gi, j = il // GU_G, il % GU_G
            wg = _load_slab(wgu[cc, :, :], (gi * 2 * GU_G + j) * HT, HT)
            wu = _load_slab(wgu[cc, :, :], (gi * 2 * GU_G + GU_G + j) * HT, HT)
            pg = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            pu = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            if it - lo < NS_HALF and C == 256:
                _gemm_tile_halves(pg, wg, xn2, HT, C)
                _gemm_tile_halves(pu, wu, xn2, HT, C)
            else:
                _gemm_tile(pg, wg, xn2, HT)
                _gemm_tile(pu, wu, xn2, HT)
            gb = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gb, src=pg)
            ub = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ub, src=pu)
            ab = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.activation(dst=ab, op=nl.silu, data=gb)
            nisa.tensor_tensor(dst=ma[:, it, :], data1=ab, data2=ub, op=nl.multiply)
            if it - m0 + 1 == GU_G or it == hi - 1:
                adma(dst=ms[:, m0:it + 1, 0:C], src=ma[:, m0:it + 1, :])
                m0 = it + 1
    nisa.core_barrier(data=ms, cores=(0, 1))
    if pid == 0:
        xdma(dst=ma[:, hi:IT, :], src=ms[:, hi:IT, 0:C], q=NS_XQM)
    else:
        xdma(dst=ma[:, 0:lo, :], src=ms[:, 0:lo, 0:C], q=NS_XQM)
    # ---- E: x_out (own half) = h1 + down(m)
    if xdst is not None:   # multi-layer NEFF: into the given scratch (the caller adds the core barrier)
        x_out = xdst
    else:
        x_out = nl.ndarray((128, HT, S), dtype=BF16, buffer=nl.shared_hbm)
    oc = nl.ndarray((128, 16, C), dtype=BF16, buffer=nl.sbuf)
    if PF_E_RING:
        ring = nl.ndarray((128, PF_E_RING, ITC * 128), dtype=BF16, buffer=nl.sbuf)
    if NS_OWNF:
        ccs = [pid, o]
        sl = 0
        for b in range(16 // NS_NB):
            pss = []
            for i in range(NS_NB):
                pss.append(nl.ndarray((128, C), dtype=F32, buffer=nl.psum))
            for cc in ccs:
                for i in range(NS_NB):
                    nt = 16 * pid + b * NS_NB + i
                    slot = sl % PF_E_RING
                    sl += 1
                    wdma(ring[:, slot, :], wd[cc, :, nt * ITC * 128:(nt + 1) * ITC * 128])
                    for k in range(ITC):
                        nisa.nc_matmul(dst=pss[i], stationary=ring[:, slot, k * 128:(k + 1) * 128],
                                       moving=ma[:, cc * ITC + k, :])
            for i in range(NS_NB):
                nt = 16 * pid + b * NS_NB + i
                db = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=db, src=pss[i])
                nisa.tensor_tensor(dst=oc[:, b * NS_NB + i, :], data1=h1[:, nt, :], data2=db, op=nl.add)
    for i in range(0 if NS_OWNF else 16):
        nt = 16 * pid + i
        ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
        for cc in range(2):
            if PF_E_RING:
                slot = (2 * i + cc) % PF_E_RING
                wdma(ring[:, slot, :], wd[cc, :, nt * ITC * 128:(nt + 1) * ITC * 128])
                for k in range(ITC):
                    nisa.nc_matmul(dst=ps, stationary=ring[:, slot, k * 128:(k + 1) * 128],
                                   moving=ma[:, cc * ITC + k, :])
            else:
                wsl = _load_slab(wd[cc, :, :], nt * ITC, ITC)
                for k in range(ITC):
                    nisa.nc_matmul(dst=ps, stationary=wsl[:, k * 128:(k + 1) * 128], moving=ma[:, cc * ITC + k, :])
        db = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=db, src=ps)
        nisa.tensor_tensor(dst=oc[:, i, :], data1=h1[:, nt, :], data2=db, op=nl.add)
    adma(dst=x_out[:, 16 * pid:16 * pid + 16, 0:C], src=oc)
    if xdst is not None:
        return x_out, oc
    return x_out


def pf_layer_tok_body(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, S, hs=None, ms=None,
                      pos=None, hcol1=False, xdst=None):
    pid = nl.program_id(0)
    SH = S // 2                # tokens per core
    C = min(256, SH)           # token chunk
    NCH = SH // C
    QB = min(128, SH)          # query block
    scale = 1.0 / (HD ** 0.5)

    cst = _consts(consts)
    ones_b = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
    nisa.memset(dst=ones_b, value=1.0)
    g1b = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=g1b, src=g1)
    g1f = nl.ndarray((128, HT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g1f, src=g1b)
    g2b = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=g2b, src=g2)
    g2f = nl.ndarray((128, HT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g2f, src=g2b)

    # ================= phase A: norm, QKV, RoPE, K/V cache writes, Q scratch
    for ch in range(NCH):
        t0 = pid * SH + ch * C
        xc = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        if PF_TOK_SYNC_X:
            for a in range(4):
                wdma(xc[:, a * 8:(a + 1) * 8, :], x[:, a * 8:(a + 1) * 8, t0:t0 + C])
            xn = _norm_fm_split(xc, g1f, ones_b, C)
        else:
            nisa.dma_copy(dst=xc, src=x[:, :, t0:t0 + C])
            xn = _norm_fm(xc, g1f, ones_b, C)
        cs = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cs, src=cosP[:, t0:t0 + C])
        sn = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sn, src=sinP[:, t0:t0 + C])
        for n in range(32 if (pos is not None and LAST_Q) else 0, 48):
            if n < 32:
                cc, ntl = n // 16, n % 16
            elif n < 40:
                cc, ntl = (n - 32) // 4, 16 + (n - 32) % 4
            else:
                cc, ntl = (n - 40) // 4, 20 + (n - 40) % 4
            wsl = _load_slab(wqkv[cc, :, :], ntl * HT, HT)
            ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            _gemm_tile(ps, wsl, xn, HT)
            yb = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=yb, src=ps)
            if n < 40:
                pr = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
                nisa.nc_matmul(dst=pr, stationary=cst[:, 0:128], moving=yb)
                a_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a_, data1=yb, data2=cs, op=nl.multiply)
                b_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=b_, data1=pr, data2=sn, op=nl.multiply)
                r_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=r_, data1=a_, data2=b_, op=nl.add)
                if n < 32:
                    q_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)   # q / sqrt(d) in bf16, as NxD
                    nisa.tensor_scalar(dst=q_, data=r_, op0=nl.multiply, operand0=QSCALE)
                    nisa.dma_copy(dst=qs[n, :, t0:t0 + C], src=q_)
                else:
                    nisa.dma_copy(dst=kc[n - 32, :, t0:t0 + C], src=r_)
                    nisa.dma_copy(dst=kq[n - 32, :, t0:t0 + C], src=r_)
            else:
                g = n - 40
                for b0 in range(0, C, 128):
                    w = min(128, C - b0)
                    pt = nl.ndarray((128, 128), dtype=F32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pt[0:w, :], stationary=yb[:, b0:b0 + w], moving=cst[:, 128:256])
                    vt = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=vt[0:w, :], src=pt[0:w, :])
                    nisa.dma_copy(dst=vc[t0 + b0:t0 + b0 + w, g * 128:(g + 1) * 128], src=vt[0:w, :])
                    nisa.dma_copy(dst=vq[g, t0 + b0:t0 + b0 + w, :], src=vt[0:w, :])

    if pos is not None and LAST_Q:   # last layer: Q and attention only for the 128 queries ending at pos
        nisa.core_barrier(data=kq, cores=(0, 1))
        nisa.core_barrier(data=vq, cores=(0, 1))
        if LH_TOKATT_TOK and hcol1:   # pf_lasth_tok: decode-style attention of the last token only
            _ns_last_tokattn(x, cosP, sinP, cst, g1, wqkv[pid, :, :], None, None, att, S, pos, kq, vq)
        else:
            _last_qattn(x, cosP, sinP, cst, g1f, ones_b, wqkv, kq, vq, att, S, pos)
        nisa.core_barrier(data=att, cores=(0, 1))
        return _pf_tail(x, wo, g2f, wgu, wd, att, hs, ms, ones_b, S, pos, barrier=False, hcol1=hcol1)
    nisa.core_barrier(data=kq, cores=(0, 1))
    nisa.core_barrier(data=vq, cores=(0, 1))
    nisa.core_barrier(data=qs, cores=(0, 1))    # attention (by heads) reads tokens of both cores

    # ================= phase B: causal attention, heads 16c.. on core c
    _attention(qs, kq, vq, att)

    nisa.core_barrier(data=att, cores=(0, 1))   # phase C reads both cores' attention outputs
    if pos is not None:
        return _pf_tail(x, wo, g2f, wgu, wd, att, hs, ms, ones_b, S, pos, barrier=False)

    # ================= phases C/D/E per chunk: o_proj + residual, norm, gate/up, SiLU*up, down + residual
    if xdst is not None:   # multi-layer NEFF: own tokens' output into the given scratch (read back by this core)
        x_out = xdst
    else:
        x_out = nl.ndarray((128, HT, S), dtype=BF16, buffer=nl.shared_hbm)
    for ch in range(NCH):
        t0 = pid * SH + ch * C
        ac = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=ac, src=att.ap(pattern=[[S, 128], [128 * S, HT], [1, C]], offset=t0))
        xc = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xc, src=x[:, :, t0:t0 + C])
        h1 = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        for nt in range(HT):
            ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            for cc in range(2):   # K halves live in the two per-core runs of the decode layout
                wsl = _load_slab(wo[cc, :, :], nt * 16, 16)
                for k in range(16):
                    nisa.nc_matmul(dst=ps, stationary=wsl[:, k * 128:(k + 1) * 128], moving=ac[:, cc * 16 + k, :])
            ob = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ob, src=ps)
            nisa.tensor_tensor(dst=h1[:, nt, :], data1=xc[:, nt, :], data2=ob, op=nl.add)
            if PF_TOK_SQ_STREAM:   # norm squares + their ones-matmul sum per finished h1 tile (same ops, same order)
                if nt == 0:
                    zps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
                sqt = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
                nisa.activation(dst=sqt, op=nl.square, data=h1[:, nt, :])
                nisa.nc_matmul(dst=zps, stationary=ones_b, moving=sqt)
        if PF_TOK_SQ_STREAM:
            xn2 = _norm_fm_from_sum(h1, g2f, zps, C)
        else:
            xn2 = _norm_fm(h1, g2f, ones_b, C)
        mT = nl.ndarray((128, IT, C), dtype=BF16, buffer=nl.sbuf)
        for it in range(IT):
            cc, il = it // ITC, it % ITC
            gi, j = il // GU_G, il % GU_G
            wg = _load_slab(wgu[cc, :, :], (gi * 2 * GU_G + j) * HT, HT)
            wu = _load_slab(wgu[cc, :, :], (gi * 2 * GU_G + GU_G + j) * HT, HT)
            pg = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            _gemm_tile(pg, wg, xn2, HT)
            pu = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            _gemm_tile(pu, wu, xn2, HT)
            gb = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gb, src=pg)
            ub = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ub, src=pu)
            ab = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.activation(dst=ab, op=nl.silu, data=gb)
            nisa.tensor_tensor(dst=mT[:, it, :], data1=ab, data2=ub, op=nl.multiply)
        for nt in range(HT):
            ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            for cc in range(2):
                wsl = _load_slab(wd[cc, :, :], nt * ITC, ITC)
                for k in range(ITC):
                    nisa.nc_matmul(dst=ps, stationary=wsl[:, k * 128:(k + 1) * 128], moving=mT[:, cc * ITC + k, :])
            db = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=db, src=ps)
            o2 = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o2, data1=h1[:, nt, :], data2=db, op=nl.add)
            nisa.dma_copy(dst=x_out[:, nt, t0:t0 + C], src=o2)
    return x_out


@nki.jit
def pf_layer(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, xa, xb, hs, S=256, CD=256):
    return pf_layer_body(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, xa, xb, hs, S,
                         min(CD, S))


@nki.jit
def pf_layer_ns(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, hs, ms, xa, S=256):
    """Head-split QKV/attention, N-split o_proj/down (bf16 activation exchanges): S <= 256."""
    return pf_layer_body(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, xa, None, hs, S,
                         S, NS=1, ms=ms)


@nki.jit
def pf_layer_tok(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, S=1024):
    """Token-split variant (each core: half of the tokens, whole weights): faster for long prompts."""
    return pf_layer_tok_body(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, S)


def wgd_own(wgu):
    return wgu[nl.program_id(0), :, :]


TAIL_SO = 0   # last-layer tail: o_proj output tiles of core 0 computed by core 1 (core 0 streams ~10% slower)
TAIL_SG = 0   # gate/up intermediate tiles (from core 0's weight run) computed by core 1
TAIL_SD = 0   # down output tiles


def _pf_tail(x, wo, g2f, wgu, wd, att, hs, ms, ones_b, S, pos, barrier, hcol1=False):
    """Last layer, phases C..E for the one token whose output is used (position pos, held on the device): the
    arithmetic of _pf_cde_nsplit / pf_layer_tok_body on a single column (each output column of a matmul is computed
    independently of the others, so the bytes are those of the full-width layer). Every output tile goes through
    HBM (hs, ms), so the split between the cores is free: core 1 takes TAIL_* tiles of core 0's share (the same
    instructions per tile: same bytes). Returns that token's hidden state h [128, HT] in the decode head's column
    layout (replaces pf_pick); hcol1: into column 1 of hs instead."""
    pid = nl.program_id(0)
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    if barrier:
        nisa.core_barrier(data=att, cores=(0, 1))
    # ---- C: h1 (own range) = x + o_proj(att)
    o0 = 0
    on = 16 - TAIL_SO
    if pid == 1:
        o0 = 16 - TAIL_SO
        on = 16 + TAIL_SO
    ac = nl.ndarray((128, HT, 1), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ac, src=att.ap(pattern=[[S, 128], [128 * S, HT], [1, 1]], offset=0, scalar_offset=preg,
                                     indirect_dim=2))
    xc = nl.ndarray((128, on, 1), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xc, src=x.ap(pattern=[[HT * S, 128], [S, on], [1, 1]], offset=o0 * S, scalar_offset=preg,
                                   indirect_dim=2))
    h1o = nl.ndarray((128, on, 1), dtype=BF16, buffer=nl.sbuf)
    for i in range(on):
        nt = o0 + i
        ps = nl.ndarray((128, 1), dtype=F32, buffer=nl.psum)
        for cc in range(2):
            wsl = _load_slab(wo[cc, :, :], nt * 16, 16)
            for k in range(16):
                nisa.nc_matmul(dst=ps, stationary=wsl[:, k * 128:(k + 1) * 128], moving=ac[:, cc * 16 + k, :])
        ob = nl.ndarray((128, 1), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ob, src=ps)
        nisa.tensor_tensor(dst=h1o[:, i, :], data1=xc[:, i, :], data2=ob, op=nl.add)
    nisa.dma_copy(dst=hs[:, o0:o0 + on, 0:1], src=h1o)
    nisa.core_barrier(data=hs, cores=(0, 1))
    # ---- D: norm(h1), gate/up tiles, SiLU * up -> ms
    h1 = nl.ndarray((128, HT, 1), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=h1, src=hs[:, :, 0:1])
    xn2 = _norm_fm(h1, g2f, ones_b, 1)
    tl = []   # (run, intermediate tile within the run)
    if pid == 0:
        for il in range(ITC - TAIL_SG):
            tl.append((0, il))
    else:
        for il in range(ITC):
            tl.append((1, il))
        for il in range(ITC - TAIL_SG, ITC):
            tl.append((0, il))
    mT = nl.ndarray((128, len(tl), 1), dtype=BF16, buffer=nl.sbuf)
    for q in range(len(tl)):
        run = tl[q][0]
        il = tl[q][1]
        gi = il // GU_G
        j = il % GU_G
        wg = _load_slab(wgu[run, :, :], (gi * 2 * GU_G + j) * HT, HT)
        wu = _load_slab(wgu[run, :, :], (gi * 2 * GU_G + GU_G + j) * HT, HT)
        pg = nl.ndarray((128, 1), dtype=F32, buffer=nl.psum)
        _gemm_tile(pg, wg, xn2, HT)
        pu = nl.ndarray((128, 1), dtype=F32, buffer=nl.psum)
        _gemm_tile(pu, wu, xn2, HT)
        gb = nl.ndarray((128, 1), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gb, src=pg)
        ub = nl.ndarray((128, 1), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ub, src=pu)
        ab = nl.ndarray((128, 1), dtype=BF16, buffer=nl.sbuf)
        nisa.activation(dst=ab, op=nl.silu, data=gb)
        nisa.tensor_tensor(dst=mT[:, q, :], data1=ab, data2=ub, op=nl.multiply)
    if pid == 0:
        nisa.dma_copy(dst=ms[:, 0:ITC - TAIL_SG, 0:1], src=mT[:, 0:ITC - TAIL_SG, :])
    else:
        nisa.dma_copy(dst=ms[:, ITC:IT, 0:1], src=mT[:, 0:ITC, :])
        if TAIL_SG > 0:
            nisa.dma_copy(dst=ms[:, ITC - TAIL_SG:ITC, 0:1], src=mT[:, ITC:ITC + TAIL_SG, :])
    nisa.core_barrier(data=ms, cores=(0, 1))
    # ---- E: h (own range) = h1 + down(m)
    ma = nl.ndarray((128, IT, 1), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ma, src=ms[:, :, 0:1])
    d0 = 0
    dn = 16 - TAIL_SD
    if pid == 1:
        d0 = 16 - TAIL_SD
        dn = 16 + TAIL_SD
    h_out = None if hcol1 else nl.ndarray((128, HT, 1), dtype=BF16, buffer=nl.shared_hbm)
    oc = nl.ndarray((128, dn, 1), dtype=BF16, buffer=nl.sbuf)
    for i in range(dn):
        nt = d0 + i
        ps = nl.ndarray((128, 1), dtype=F32, buffer=nl.psum)
        for cc in range(2):
            wsl = _load_slab(wd[cc, :, :], nt * ITC, ITC)
            for k in range(ITC):
                nisa.nc_matmul(dst=ps, stationary=wsl[:, k * 128:(k + 1) * 128],
                               moving=ma[:, cc * ITC + k, :])
        db = nl.ndarray((128, 1), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=db, src=ps)
        nisa.tensor_tensor(dst=oc[:, i, :], data1=h1[:, nt, :], data2=db, op=nl.add)
    if hcol1:   # fused head (pf_lasth_*): the halves meet in column 1 of hs
        nisa.dma_copy(dst=hs[:, d0:d0 + dn, 1:2], src=oc)
        return None
    nisa.dma_copy(dst=h_out[:, d0:d0 + dn, :], src=oc)
    return h_out


@nki.jit
def pf_last_ns(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, hs, ms, pos, S=256):
    """Last layer, S <= 256: pf_layer_ns phases A/B, then the tail for the last prompt token only."""
    return pf_layer_body(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, None, None, hs, S,
                         S, NS=1, ms=ms, pos=pos)


@nki.jit
def pf_last_tok(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, hs, ms, pos, S=1024):
    """Last layer, S >= 1024: pf_layer_tok phases A/B, then the tail for the last prompt token only."""
    return pf_layer_tok_body(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, S, hs=hs,
                             ms=ms, pos=pos)


@nki.jit
def pf_embed(toks, embed, consts, S=256):
    """x [128, HT, S] feature-major = embed[toks[0, s]] for s < S."""
    pid = nl.program_id(0)
    SH = S // 2
    cst = _consts(consts)
    x = nl.ndarray((128, HT, S), dtype=BF16, buffer=nl.shared_hbm)
    for b0 in range(0, SH, 128):
        w = min(128, SH - b0)
        t0 = pid * SH + b0
        idx = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx[0:w, :], src=toks.ap(pattern=[[1, w], [1, 1]], offset=t0))
        rows = nl.ndarray((128, H), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows[0:w, :], src=embed.ap(pattern=[[H, w], [1, H]], offset=0, vector_offset=idx[0:w, :],
                                                       indirect_dim=0))
        xs = nl.ndarray((128, HT, 128), dtype=BF16, buffer=nl.sbuf)
        for kt in range(HT):
            pt = nl.ndarray((128, 128), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(dst=pt[:, 0:w], stationary=rows[0:w, kt * 128:(kt + 1) * 128], moving=cst[0:w, 128:128 + w])
            nisa.tensor_copy(dst=xs[:, kt, 0:w], src=pt[:, 0:w])
        nisa.dma_copy(dst=x[:, :, t0:t0 + w], src=xs[:, :, 0:w])
    return x


@nki.jit
def pf_pick(x, pos):
    """h [128, HT] = x[:, :, pos] (column layout for the decode head)."""
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    S = x.shape[2]
    hb = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=hb, src=x.ap(pattern=[[HT * S, 128], [S, HT]], offset=0, scalar_offset=preg, indirect_dim=2))
    h = nl.ndarray((128, HT), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=h, src=hb)
    return h


# ===================================================================================================================
# Pipelined token-split layer (S >= 1024): attention by query chunks, interleaved with the previous chunk's MLP.
# Core c owns the 256-token chunks 4p (+0, +3) for c = 0 and 4p (+1, +2) for c = 1 (equal causal attention work);
# phase A (QKV) for its own chunks; barrier on K/V; then, per own chunk i, the o_proj / MLP steps of chunk i are
# emitted interleaved with the attention stages of chunk i+1 (all 32 heads), so the engines' in-order instruction
# streams overlap the attention's Scalar / Vector work with the MLP's TensorE work. Same arithmetic per chunk as
# pf_layer_tok_body and as nkilib attention (pattn): the same bytes.

PIPE_C = 256
PIPE_SEQ = 0     # 1: no interleaving (attention of chunk i+1 after the MLP of chunk i), for measurements


def _own_chunks(pid, NCT):
    r = []
    for p in range(NCT // 4):
        if pid == 0:
            r.append(4 * p)
            r.append(4 * p + 3)
        else:
            r.append(4 * p + 1)
            r.append(4 * p + 2)
    return r


def _phase_a_chunk(x, cosP, sinP, cst, g1f, ones_b, wqkv, kc, vc, qs, kq, vq, t0, C, at=None):
    """pf_layer_tok_body phase A for the tokens t0 .. t0+C-1 (all 32 q heads, 8 kv heads).
    at = (as_, ac_, att, recip, j): the attention steps of chunk j are emitted spread over the 48 QKV tiles
    (PIPE_A2); step 0 (q and K/V loads) was emitted by the caller."""
    done = 0
    if at is not None:
        done = 1
    xc = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
    for a in range(4):
        wdma(xc[:, a * 8:(a + 1) * 8, :], x[:, a * 8:(a + 1) * 8, t0:t0 + C])
    xn = _norm_fm_split(xc, g1f, ones_b, C)
    cs = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs, src=cosP[:, t0:t0 + C])
    sn = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sn, src=sinP[:, t0:t0 + C])
    for n in range(48):
        if n < 32:
            cc, ntl = n // 16, n % 16
        elif n < 40:
            cc, ntl = (n - 32) // 4, 16 + (n - 32) % 4
        else:
            cc, ntl = (n - 40) // 4, 20 + (n - 40) % 4
        wsl = _load_slab(wqkv[cc, :, :], ntl * HT, HT)
        ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
        _gemm_tile(ps, wsl, xn, HT)
        yb = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=yb, src=ps)
        if n < 40:
            pr = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(dst=pr, stationary=cst[:, 0:128], moving=yb)
            a_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=a_, data1=yb, data2=cs, op=nl.multiply)
            b_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=b_, data1=pr, data2=sn, op=nl.multiply)
            r_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r_, data1=a_, data2=b_, op=nl.add)
            if n < 32:
                q_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)   # q / sqrt(d) in bf16, as NxD
                nisa.tensor_scalar(dst=q_, data=r_, op0=nl.multiply, operand0=QSCALE)
                nisa.dma_copy(dst=qs[n, :, t0:t0 + C], src=q_)
            else:
                nisa.dma_copy(dst=kc[n - 32, :, t0:t0 + C], src=r_)
                nisa.dma_copy(dst=kq[n - 32, :, t0:t0 + C], src=r_)
        else:
            g = n - 40
            for b0 in range(0, C, 128):
                w = min(128, C - b0)
                pt = nl.ndarray((128, 128), dtype=F32, buffer=nl.psum)
                nisa.nc_matmul(dst=pt[0:w, :], stationary=yb[:, b0:b0 + w], moving=cst[:, 128:256])
                vt = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=vt[0:w, :], src=pt[0:w, :])
                nisa.dma_copy(dst=vc[t0 + b0:t0 + b0 + w, g * 128:(g + 1) * 128], src=vt[0:w, :])
                nisa.dma_copy(dst=vq[g, t0 + b0:t0 + b0 + w, :], src=vt[0:w, :])
        if at is not None:
            upto = ((n + 1) * ATT_NSTEP) // 48
            for k in range(done, upto):
                _at_step(k, at[0], at[1], qs, kq, vq, at[2], at[3], at[4], C, True)
            done = upto


PIPE_A2 = 0   # S = 1024: phase A of the first own chunk, K/V barrier, then the first own chunk's attention emitted
              # beside phase A of the second own chunk (its keys are chunks 0..1, already exchanged), second barrier


PC_NSTEP = 1 + 32 + 1 + IT + 32   # phase C steps per chunk: loads, o_proj tiles, norm, gate/up tiles, down tiles


def _pc_step(s, cs_, x, wo, g2f, ones_b, wgu, wd, att, x_out, S, t0, C):
    """Step s of pf_layer_tok_body's phases C/D/E for the chunk at t0 (state list cs_ = [ac, h1, xn2, mT])."""
    if s == 0:
        ac = cs_[0]   # the chunk's attention output, in SBUF (from _at_step)
        h1 = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        mT = nl.ndarray((128, IT, C), dtype=BF16, buffer=nl.sbuf)
        cs_.append(h1)
        cs_.append(mT)
        if PF_PIPE_SQ_STREAM:
            cs_.append(nl.ndarray((128, C), dtype=F32, buffer=nl.psum))   # column sums of the norm squares
    elif s <= 32:
        nt = s - 1
        ac = cs_[0]
        h1 = cs_[1]
        xc = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xc, src=x[:, nt, t0:t0 + C])
        ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
        for cc in range(2):   # K halves live in the two per-core runs of the decode layout
            wsl = _load_slab(wo[cc, :, :], nt * 16, 16)
            for k in range(16):
                nisa.nc_matmul(dst=ps, stationary=wsl[:, k * 128:(k + 1) * 128], moving=ac[:, cc * 16 + k, :])
        ob = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ob, src=ps)
        nisa.tensor_tensor(dst=h1[:, nt, :], data1=xc, data2=ob, op=nl.add)
        if PF_PIPE_SQ_STREAM:   # norm squares + their ones-matmul sum per finished h1 tile (same ops, same order)
            sqt = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.activation(dst=sqt, op=nl.square, data=h1[:, nt, :])
            nisa.nc_matmul(dst=cs_[3], stationary=ones_b, moving=sqt)
    elif s == 33:
        if PF_PIPE_SQ_STREAM:
            cs_[3] = _norm_fm_from_sum(cs_[1], g2f, cs_[3], C)
        else:
            cs_.append(_norm_fm(cs_[1], g2f, ones_b, C))
    elif s < 34 + IT:
        it = s - 34
        xn2 = cs_[3]
        mT = cs_[2]
        cc, il = it // ITC, it % ITC
        gi, j = il // GU_G, il % GU_G
        wg = _load_slab(wgu[cc, :, :], (gi * 2 * GU_G + j) * HT, HT)
        wu = _load_slab(wgu[cc, :, :], (gi * 2 * GU_G + GU_G + j) * HT, HT)
        pg = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
        _gemm_tile(pg, wg, xn2, HT)
        pu = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
        _gemm_tile(pu, wu, xn2, HT)
        gb = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gb, src=pg)
        ub = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ub, src=pu)
        ab = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
        nisa.activation(dst=ab, op=nl.silu, data=gb)
        nisa.tensor_tensor(dst=mT[:, it, :], data1=ab, data2=ub, op=nl.multiply)
    else:
        nt = s - 34 - IT
        h1 = cs_[1]
        mT = cs_[2]
        ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
        for cc in range(2):
            wsl = _load_slab(wd[cc, :, :], nt * ITC, ITC)
            for k in range(ITC):
                nisa.nc_matmul(dst=ps, stationary=wsl[:, k * 128:(k + 1) * 128], moving=mT[:, cc * ITC + k, :])
        db = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=db, src=ps)
        o2 = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o2, data1=h1[:, nt, :], data2=db, op=nl.add)
        nisa.dma_copy(dst=x_out[:, nt, t0:t0 + C], src=o2)


ATT_D_EXP = 1          # pipeline distances (in units) of the attention stages behind QK
ATT_D_TP = 2
ATT_D_PV = 4
ATT_D_WB = 5
ATT_KV_AHEAD = 4       # K/V of a kv head loaded this many units before its first QK
ATT_NSTEP = 64 + ATT_D_WB    # attention steps per chunk: 32 heads x 2 query groups, plus the pipeline drain
PIPE_PE_TP = True      # P^T on the TensorE (measured: same time as DMA transposes at S = 1024, fewer waits)
PIPE_WIN = 0           # phase C steps beside which attention steps go: 0 o_proj + norm + down, 1 down only


def _at_unit(u, j, gpc):
    kvh = u // (4 * gpc)
    h = 4 * kvh + (u % (4 * gpc)) // gpc
    grp = gpc * j + u % gpc
    return kvh, h, grp


def _at_step(k, as_, ac_, qs, kq, vq, att, recip, j, C, allkv=False):
    """Step k of the attention of chunk j (query groups 2j, 2j+1 when C = 256), units u = kv-head-major
    (kvh = u // 8, head 4*kvh + (u % 8) // 2, group 2j + u % 2): K/V loads ahead, QK of unit k, exp of
    k - ATT_D_EXP, P^T of k - ATT_D_TP, PV of k - ATT_D_PV, store of k - ATT_D_WB.
    as_ = [kv buffers, unit states, q chunk]."""
    gpc = C // 128
    upk = 4 * gpc
    kvl = as_[0]
    sts = as_[1]
    nkeys = (j + 1) * C
    if k == 0:   # q of the whole chunk (32 heads) in one DMA: [128 d, 32 h, C]
        qc = nl.ndarray((128, 32, C), dtype=BF16, buffer=nl.sbuf)
        S = qs.shape[2]
        nisa.dma_copy(dst=qc, src=qs.ap(pattern=[[S, 128], [128 * S, 32], [1, C]], offset=j * C),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        as_[2].append(qc)
        acb = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)   # attention output of the chunk (o_proj input)
        as_[2].append(acb)
        for kvh in range(8):
            if kvh * upk - ATT_KV_AHEAD <= 0 or allkv:   # allkv: all K/V loaded before any later kq/vq write
                k_sb, v_sb = pattn.load_kv(kq, vq, kvh, nkeys)
                kvl.append(k_sb)
                kvl.append(v_sb)
    elif not allkv:
        for kvh in range(8):
            if kvh * upk - ATT_KV_AHEAD == k:
                k_sb, v_sb = pattn.load_kv(kq, vq, kvh, nkeys)
                kvl.append(k_sb)
                kvl.append(v_sb)
    qc = as_[2][0]
    if k < 64:
        kvh, h, grp = _at_unit(k, j, gpc)
        sts.append(pattn.stage_qk(ac_, qs, h, grp, kvl[2 * kvh], qc[:, h, (grp - gpc * j) * 128:(grp - gpc * j + 1) * 128]))
    u = k - ATT_D_EXP
    if u >= 0 and u < 64:
        pattn.stage_exp(ac_, sts[u], ATT_D_TP == ATT_D_EXP, PIPE_PE_TP)
    u = k - ATT_D_TP
    if u >= 0 and u < 64 and ATT_D_TP != ATT_D_EXP:
        pattn.stage_tp(ac_, sts[u], PIPE_PE_TP)
    u = k - ATT_D_PV
    if u >= 0 and u < 64:
        kvh, h, grp = _at_unit(u, j, gpc)
        pattn.stage_pv(ac_, sts[u], kvl[2 * kvh + 1])
    u = k - ATT_D_WB
    if u >= 0 and u < 64:
        kvh, h, grp = _at_unit(u, j, gpc)
        gl = grp - gpc * j
        pattn.stage_wb(ac_, sts[u], att, recip, as_[2][1][:, h, gl * 128:(gl + 1) * 128])


def _pc_weight(s):
    """TensorE work (matmuls) of phase C step s, if attention steps may go beside it, else 0."""
    if s == 0:
        return 0
    if s <= 32:
        return 32 if PIPE_WIN == 0 else 0
    if s == 33:
        return 32 if PIPE_WIN == 0 else 0
    if s < 34 + IT:
        return 0
    return 112


def pf_layer_pipe_body(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, S, xdst=None):
    pid = nl.program_id(0)
    C = PIPE_C
    NCT = S // C
    own = _own_chunks(pid, NCT)
    n = len(own)

    cst = _consts(consts)
    ones_b = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
    nisa.memset(dst=ones_b, value=1.0)
    g1b = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=g1b, src=g1)
    g1f = nl.ndarray((128, HT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g1f, src=g1b)
    g2b = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=g2b, src=g2)
    g2f = nl.ndarray((128, HT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g2f, src=g2b)

    ac_ = pattn.attn_consts(S)
    recip = nl.ndarray((128, S // 128), dtype=F32, buffer=nl.sbuf)
    if xdst is not None:   # multi-layer NEFF: own tokens' output into the given scratch (read back by this core)
        x_out = xdst
    else:
        x_out = nl.ndarray((128, HT, S), dtype=BF16, buffer=nl.shared_hbm)
    as0 = [[], [], []]
    if PIPE_A2 and n == 2 and own[0] <= 1 and own[1] >= 2:
        _phase_a_chunk(x, cosP, sinP, cst, g1f, ones_b, wqkv, kc, vc, qs, kq, vq, own[0] * C, C)
        nisa.core_barrier(data=kq, cores=(0, 1))
        nisa.core_barrier(data=vq, cores=(0, 1))
        _at_step(0, as0, ac_, qs, kq, vq, att, recip, own[0], C, True)   # q + all K/V before the next writes
        _phase_a_chunk(x, cosP, sinP, cst, g1f, ones_b, wqkv, kc, vc, qs, kq, vq, own[1] * C, C,
                       (as0, ac_, att, recip, own[0]))
        nisa.core_barrier(data=kq, cores=(0, 1))
        nisa.core_barrier(data=vq, cores=(0, 1))
    else:
        # ================= phase A for the own chunks
        for i in range(n):
            _phase_a_chunk(x, cosP, sinP, cst, g1f, ones_b, wqkv, kc, vc, qs, kq, vq, own[i] * C, C)
        nisa.core_barrier(data=kq, cores=(0, 1))
        nisa.core_barrier(data=vq, cores=(0, 1))
        # attention of the first own chunk on its own
        for k in range(ATT_NSTEP):
            _at_step(k, as0, ac_, qs, kq, vq, att, recip, own[0], C)
    acbs = [as0[2][1]]
    for i in range(n):
        t0 = own[i] * C
        cs_ = [acbs[i]]
        if i + 1 < n and PIPE_SEQ:
            for s in range(PC_NSTEP):
                _pc_step(s, cs_, x, wo, g2f, ones_b, wgu, wd, att, x_out, S, t0, C)
            asn = [[], [], []]
            for k in range(ATT_NSTEP):
                _at_step(k, asn, ac_, qs, kq, vq, att, recip, own[i + 1], C)
            acbs.append(asn[2][1])
        elif i + 1 < n:
            asn = [[], [], []]
            done = 0
            wsum = 0
            wtot = 0
            for s in range(PC_NSTEP):
                wtot += _pc_weight(s)
            for s in range(PC_NSTEP):
                _pc_step(s, cs_, x, wo, g2f, ones_b, wgu, wd, att, x_out, S, t0, C)
                # attention steps only beside o_proj / down (exp and SiLU need different activation tables: a
                # table load per switch, ~1.3 us), spread by the TensorE work of those steps
                wsum += _pc_weight(s)
                upto = (wsum * ATT_NSTEP) // wtot
                for k in range(done, upto):
                    _at_step(k, asn, ac_, qs, kq, vq, att, recip, own[i + 1], C)
                done = upto
            acbs.append(asn[2][1])
        else:
            for s in range(PC_NSTEP):
                _pc_step(s, cs_, x, wo, g2f, ones_b, wgu, wd, att, x_out, S, t0, C)
    return x_out


@nki.jit
def pf_layer_pipe(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, S=1024):
    return pf_layer_pipe_body(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, S)


# ===================================================================================================================
# S <= 256, phases A + B pipelined: QKV in kv-group order (k_g, v_g, q heads 4g..4g+3 of this core), the attention
# units of group g (from SBUF: no q/k/v scratch round trip) interleaved with the QKV GEMMs of group g+1.
# Same arithmetic as phases A / B of pf_layer_body (QKV tiles are independent; attention = pattn, nkilib's bytes).

NS_PIPE = 1
NS_D = (1, 2, 3, 4)    # unit distances of exp, P^T, PV, store behind QK
NS_RAMP = 6            # QKV tiles of group g+1 over which the attention units of group g are spread
MULTI_PRE = 6          # pf_multi: QKV tiles accumulated over the own K half before the other half is loaded
MULTI_ORDER = [16, 20, 0, 1, 2, 3]   # = the first entries of _ns_phase_ab's tile order (k_0, v_0, q 0..3)
NS_ATT2 = 1            # S <= 256: the last kv group's attention heads stored to qs (own core barrier and load), so
                       # o_proj's first 28 K tiles do not wait for its store latency
NS_PE_TP = True        # P^T on the TensorE (phase A is weight-stream bound) instead of DMA transposes


def _ns_phase_ab(x, cosP, sinP, cst, g1f, ones_b, wq_c, kc, vc, att, S, acb=None, xsb=None, xown=None, att2=None):
    pid = nl.program_id(0)
    o = 1 - pid
    C = S
    NG = S // 128
    UPG = 4 * NG                 # attention units (head, query group) per kv group
    NU = 4 * UPG
    npre = 0
    if xown is not None and NS_POSTR:   # pf_multi: own half of x in SBUF; the first MULTI_PRE QKV tiles accumulate
        # the own K half while the other core's half is loaded (fp32 order on core 1: own half first)
        npre = MULTI_PRE
        xn = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        for k in range(16):
            kt = 16 * pid + k
            if kt % 2 == 0:
                nisa.tensor_scalar(dst=xn[:, kt, :], data=xown[:, k, :], op0=nl.multiply, operand0=g1f[:, kt:kt + 1])
            else:
                nisa.activation(dst=xn[:, kt, :], op=nl.copy, data=xown[:, k, :], scale=g1f[:, kt:kt + 1])
        pre_ps = []
        pre_w = []
        for m in range(npre):
            n = MULTI_ORDER[m]
            wsl = _load_slab(wq_c, n * HT, HT)
            ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            for k in range(16):
                kt = 16 * pid + k
                nisa.nc_matmul(dst=ps, stationary=wsl[:, kt * 128:(kt + 1) * 128], moving=xn[:, kt, :])
            pre_ps.append(ps)
            pre_w.append(wsl)
        xoth = nl.ndarray((128, 16, C), dtype=BF16, buffer=nl.sbuf)
        for a in range(2):
            wdma(xoth[:, a * 8:(a + 1) * 8, :], x[:, 16 * o + a * 8:16 * o + (a + 1) * 8, 0:C])
        for k in range(16):
            kt = 16 * o + k
            if kt % 2 == 0:
                nisa.tensor_scalar(dst=xn[:, kt, :], data=xoth[:, k, :], op0=nl.multiply, operand0=g1f[:, kt:kt + 1])
            else:
                nisa.activation(dst=xn[:, kt, :], op=nl.copy, data=xoth[:, k, :], scale=g1f[:, kt:kt + 1])
        sq = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        nisa.activation(dst=sq[:, 0:16, :], op=nl.square, data=xown)
        nisa.tensor_tensor(dst=sq[:, 16:32, :], data1=xoth, data2=xoth, op=nl.multiply)
        pss = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
        for kt in range(HT):
            nisa.nc_matmul(dst=pss, stationary=ones_b, moving=sq[:, kt, :])
        t_ = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=t_, data=pss, op0=nl.multiply, operand0=1.0 / H, op1=nl.add, operand1=EPS)
        rs1 = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rs1, data=t_, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
        xc = None
    elif xsb is not None:   # layer 0 with the embedding gather fused (pf_first_ns): x already in SBUF
        xc = xsb
    else:
        xc = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        for a in range(4):
            wdma(xc[:, a * 8:(a + 1) * 8, :], x[:, a * 8:(a + 1) * 8, 0:C])
    if npre:
        pass
    elif NS_POSTR:   # GEMM on bf16(x * g); the per-token rstd applied to the fp32 GEMM result (off the critical path)
        rs1 = None
        xn = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
        _scale_g(xn, xc, g1f, 0, HT)
        rs1 = _rstd_fm(xc, ones_b, C, NS_DVSQ)
    else:
        rs1 = None
        xn = _norm_fm_split(xc, g1f, ones_b, C, ngps=NS_NGPS, halves=NS_HALF > 0, dvsq=NS_DVSQ)
    cs = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs, src=cosP[:, 0:C])
    sn = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sn, src=sinP[:, 0:C])
    ac_ = pattn.attn_consts(S)
    recip = nl.ndarray((128, NG), dtype=F32, buffer=nl.sbuf)
    ks = []
    vs = []
    qh = []
    sts = []
    st_k = [0]
    order = []
    for g in range(4):
        order.append(16 + g)
        order.append(20 + g)
        for i in range(4):
            order.append(4 * g + i)
    for m in range(QKV_NT):
        n = order[m]
        if m < npre:   # own K half already accumulated: the other half
            wsl = pre_w[m]
            ps = pre_ps[m]
            for k in range(16):
                kt = 16 * o + k
                nisa.nc_matmul(dst=ps, stationary=wsl[:, kt * 128:(kt + 1) * 128], moving=xn[:, kt, :])
        else:
            wsl = _load_slab(wq_c, n * HT, HT)
            ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
        if m < npre:
            pass
        elif m < NS_HALF and C == 256:
            _gemm_tile_halves(ps, wsl, xn, HT, C)
        else:
            _gemm_tile(ps, wsl, xn, HT)
        yb = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
        if rs1 is not None:
            nisa.tensor_tensor(dst=yb, data1=ps, data2=rs1, op=nl.multiply)
        else:
            nisa.tensor_copy(dst=yb, src=ps)
        if n < 20:
            pr = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(dst=pr, stationary=cst[:, 0:128], moving=yb)
            a_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=a_, data1=yb, data2=cs, op=nl.multiply)
            b_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=b_, data1=pr, data2=sn, op=nl.multiply)
            r_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r_, data1=a_, data2=b_, op=nl.add)
            if n < 16:
                q_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)   # q / sqrt(d) in bf16, as NxD
                nisa.tensor_scalar(dst=q_, data=r_, op0=nl.multiply, operand0=QSCALE)
                qh.append(q_)
            else:
                nisa.dma_copy(dst=kc[4 * pid + n - 16, :, 0:C], src=r_)
                ks.append(r_)
        else:
            G = 4 * pid + n - 20
            v_sb = nl.ndarray((128, NG, 128), dtype=BF16, buffer=nl.sbuf)
            for b in range(NG):
                pt = nl.ndarray((128, 128), dtype=F32, buffer=nl.psum)
                nisa.nc_matmul(dst=pt, stationary=yb[:, b * 128:(b + 1) * 128], moving=cst[:, 128:256])
                nisa.tensor_copy(dst=v_sb[:, b, :], src=pt)
                nisa.dma_copy(dst=vc[b * 128:(b + 1) * 128, G * 128:(G + 1) * 128], src=v_sb[:, b, :])
            vs.append(v_sb)
        # attention steps for the kv groups whose QKV is complete, spread over the next group's 6 tiles
        gdone = (m + 1) // 6
        avail = UPG * gdone
        target = (UPG * (m + 1 - 6) + NS_RAMP - 1) // NS_RAMP if m >= 6 else 0
        if m == QKV_NT - 1:
            target = NU + NS_D[3]
        while st_k[0] < target and (st_k[0] < avail or st_k[0] >= NU):
            _ns_att_step(st_k[0], ks, vs, qh, sts, ac_, att, recip, S, acb, att2)
            st_k[0] = st_k[0] + 1


def _ns_att_step(k, ks, vs, qh, sts, ac_, att, recip, S, acb=None, att2=None):
    pid = nl.program_id(0)
    NG = S // 128
    NU = 16 * NG
    if k < NU:
        hl = k // NG
        grp = k % NG
        sts.append(pattn.stage_qk(ac_, None, 16 * pid + hl, grp, ks[hl // 4], qh[hl][:, grp * 128:(grp + 1) * 128]))
    u = k - NS_D[0]
    if u >= 0 and u < NU:
        pattn.stage_exp(ac_, sts[u], False)
    u = k - NS_D[1]
    if u >= 0 and u < NU:
        pattn.stage_tp(ac_, sts[u], NS_PE_TP)
    u = k - NS_D[2]
    if u >= 0 and u < NU:
        pattn.stage_pv(ac_, sts[u], vs[(u // NG) // 4])
    u = k - NS_D[3]
    if u >= 0 and u < NU:
        hl = u // NG
        grp = u % NG
        if acb is None:
            ob = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
            pattn.stage_wb(ac_, sts[u], att, recip, ob)
            adma(dst=att[16 * pid + hl, :, grp * 128:(grp + 1) * 128], src=ob)
        else:   # into the SBUF copy; one store per finished kv group (4 heads)
            pattn.stage_wb(ac_, sts[u], att, recip, acb[:, 16 * pid + hl, grp * 128:(grp + 1) * 128])
            if u % (4 * NG) == 4 * NG - 1:
                h0 = 16 * pid + 4 * (u // (4 * NG))
                at_ = att
                if att2 is not None and u // (4 * NG) == 3:   # NS_ATT2: the last kv group into its own tensor
                    at_ = att2
                sdma(dst=at_.ap(pattern=[[S, 128], [128 * S, 4], [1, S]], offset=h0 * 128 * S),
                     src=acb[:, h0:h0 + 4, :])


NS_PSX = 1    # S <= 256: the norm before the MLP from exchanged partial sums of squares (xa scratch): each core
              # normalizes its own half of h1 and starts gate/up on it (own K half first) while the other half arrives


NS_XN2X = 1   # S <= 256 (NS_POSTR + NS_PSX): each core scales its own h1 half by g2 as o_proj produces it and stores
              # that (the other core's gate/up operand) to hs: no scaling after the exchange (same bytes)


def _ns_d_psx(h1, g2f, ones_b, wgu, hs, ms, xa, ma, lo, hi, S, xn2x=None):
    pid = nl.program_id(0)
    o = 1 - pid
    C = S
    sq2 = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
    for a in range(2):
        a2 = 2 * pid + a
        if NS_DVSQ and a == 1:
            nisa.tensor_tensor(dst=sq2[:, a2 * 8:(a2 + 1) * 8, :], data1=h1[:, a2 * 8:(a2 + 1) * 8, :],
                               data2=h1[:, a2 * 8:(a2 + 1) * 8, :], op=nl.multiply)
        else:
            nisa.activation(dst=sq2[:, a2 * 8:(a2 + 1) * 8, :], op=nl.square, data=h1[:, a2 * 8:(a2 + 1) * 8, :])
    psA = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
    for kt in range(16 * pid, 16 * pid + 16):
        nisa.nc_matmul(dst=psA, stationary=ones_b, moving=sq2[:, kt, :])
    pA = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pA, src=psA)
    if xn2x is not None:
        xn2 = xn2x
    else:
        xn2 = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
    if NS_POSTR and xn2x is None:
        _scale_g(xn2, h1, g2f, 16 * pid, 16 * pid + 16)
    stq(dst=xa[pid, 0:1, 0, 0:C], src=pA[0:1, :], q=NS_STQX)
    rstd = None
    if not NS_POSTR:
        rstd = _ns_psx_exch(xa, pA, h1, xn2, g2f, hs, S)
    m0 = lo
    NBG = NS_NBG
    for b0 in range(lo, hi, NBG):
        b1 = min(hi, b0 + NBG)
        pgs = []
        pus = []
        for it in range(b0, b1):
            pgs.append(nl.ndarray((128, C), dtype=F32, buffer=nl.psum))
            pus.append(nl.ndarray((128, C), dtype=F32, buffer=nl.psum))
        for kh in (pid, o):   # K half: own features first
            if kh == o and rstd is None:   # NS_POSTR: exchanges placed after the first own-half matmuls
                rstd = _ns_psx_exch(xa, pA, h1, xn2, g2f, hs, S, xn2x is not None)
            for it in range(b0, b1):
                cc, il = it // ITC, it % ITC
                gi, j = il // GU_G, il % GU_G
                wg = _load_slab(wgu[cc, :, :], (gi * 2 * GU_G + j) * HT + 16 * kh, 16)
                wu = _load_slab(wgu[cc, :, :], (gi * 2 * GU_G + GU_G + j) * HT + 16 * kh, 16)
                for k in range(16):
                    nisa.nc_matmul(dst=pgs[it - b0], stationary=wg[:, k * 128:(k + 1) * 128], moving=xn2[:, 16 * kh + k, :])
                for k in range(16):
                    nisa.nc_matmul(dst=pus[it - b0], stationary=wu[:, k * 128:(k + 1) * 128], moving=xn2[:, 16 * kh + k, :])
        for it in range(b0, b1):
            gb = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            ub = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            if NS_POSTR:
                nisa.tensor_tensor(dst=gb, data1=pgs[it - b0], data2=rstd, op=nl.multiply)
                nisa.tensor_tensor(dst=ub, data1=pus[it - b0], data2=rstd, op=nl.multiply)
            else:
                nisa.tensor_copy(dst=gb, src=pgs[it - b0])
                nisa.tensor_copy(dst=ub, src=pus[it - b0])
            ab = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.activation(dst=ab, op=nl.silu, data=gb)
            nisa.tensor_tensor(dst=ma[:, it, :], data1=ab, data2=ub, op=nl.multiply)
            if it - m0 + 1 == GU_G or it == hi - 1:
                stq(dst=ms[:, m0:it + 1, 0:C], src=ma[:, m0:it + 1, :])
                m0 = it + 1


# ===================================================================================================================
# Last layer, S >= 1024: only the last prompt token's output is used, so Q and attention are computed for the 128
# queries ending at pos (a window at a dynamic offset) instead of all S; K / V for all tokens as before. Per query row
# the arithmetic is that of pf_layer_tok_body / pattn: the same q (matmul columns are independent), the same scores,
# per-tile maxima and exp values; key tiles past the row's causal bound are fully masked (exp = 0), the diagonal tile
# is exp'ed over all 512 keys (the extra keys are masked to 0).

LAST_Q = 1


def _last_qattn(x, cosP, sinP, cst, g1f, ones_b, wqkv, kq, vq, att, S, pos):
    pid = nl.program_id(0)
    W = 128
    FMIN = pattn.FMIN
    KT = min(pattn.KT, S)   # S = 256: one 256-key tile (S >= 1024: unchanged)
    LT = min(pattn.LT, S)
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    pw_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)   # window start pos - 127
    nisa.tensor_scalar(dst=pw_sb, data=p_sb, op0=nl.subtract, operand0=W - 1)
    pwreg = nisa.register_alloc()
    nisa.register_load(pwreg, pw_sb)
    # causal bound of window row p: query position pos - 127 + p (fp32, per partition)
    p128 = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p128, src=pos.ap(pattern=[[0, 128], [1, 1]], offset=0))
    pf = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pf, src=p128)
    io = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    ub0 = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ub0, data1=pf, data2=io, op=nl.add)
    ub = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ub, data=ub0, op0=nl.add, operand0=-float(W - 1))
    zero = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=zero, value=0.0)
    # window x, norm, own q heads, RoPE (window positions)
    xc = nl.ndarray((128, HT, W), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xc, src=x.ap(pattern=[[HT * S, 128], [S, HT], [1, W]], offset=0, scalar_offset=pwreg,
                                   indirect_dim=2))
    xn = _norm_fm_split(xc, g1f, ones_b, W)
    TM = cosP.shape[1]
    cs = nl.ndarray((128, W), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs, src=cosP.ap(pattern=[[TM, 128], [1, W]], offset=0, scalar_offset=pwreg, indirect_dim=1))
    sn = nl.ndarray((128, W), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sn, src=sinP.ap(pattern=[[TM, 128], [1, W]], offset=0, scalar_offset=pwreg, indirect_dim=1))
    wq_c = wqkv[pid, :, :]
    qh = []
    for n in range(16):
        wsl = _load_slab(wq_c, n * HT, HT)
        ps = nl.ndarray((128, W), dtype=F32, buffer=nl.psum)
        _gemm_tile(ps, wsl, xn, HT)
        yb = nl.ndarray((128, W), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=yb, src=ps)
        pr = nl.ndarray((128, W), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=pr, stationary=cst[:, 0:128], moving=yb)
        a_ = nl.ndarray((128, W), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=a_, data1=yb, data2=cs, op=nl.multiply)
        b_ = nl.ndarray((128, W), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=b_, data1=pr, data2=sn, op=nl.multiply)
        r_ = nl.ndarray((128, W), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r_, data1=a_, data2=b_, op=nl.add)
        q_ = nl.ndarray((128, W), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=q_, data=r_, op0=nl.multiply, operand0=QSCALE)
        qh.append(q_)
    nkt = S // KT
    nlt = (S + LT - 1) // LT
    for hl in range(16):
        h = 16 * pid + hl
        if hl % 4 == 0:
            k_sb, v_sb = pattn.load_kv(kq, vq, 4 * pid + hl // 4, S)
        pmax = nl.ndarray((128, nkt), dtype=F32, buffer=nl.sbuf)
        nisa.memset(dst=pmax, value=FMIN)
        masked = nl.ndarray((128, S), dtype=F32, buffer=nl.sbuf)
        for t in range(nkt):
            k0 = t * KT
            ps = nl.ndarray((128, KT), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps, stationary=qh[hl], moving=k_sb[:, k0:k0 + KT])
            nisa.range_select(dst=masked[:, k0:k0 + KT], on_true_tile=ps, on_false_value=FMIN,
                              comp_op0=nl.greater_equal, comp_op1=nl.less_equal, bound0=zero[:, 0:1],
                              bound1=ub[:, 0:1], reduce_op=nl.maximum, reduce_res=pmax[:, t:t + 1],
                              reduce_cmd=nisa.reduce_cmd.reset_reduce, range_start=k0)
        smax = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=smax[:, 0:1], op=nl.maximum, data=pmax, axis=1, negate=True)
        rmax = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rmax, src=smax)
        psums = nl.ndarray((128, nkt), dtype=F32, buffer=nl.sbuf)
        nisa.memset(dst=psums, value=0.0)
        e_sb = nl.ndarray((128, S), dtype=BF16, buffer=nl.sbuf)
        for t in range(nkt):
            k0 = t * KT
            nisa.activation_reduce(dst=e_sb[:, k0:k0 + KT], op=nl.exp, data=masked[:, k0:k0 + KT], reduce_op=nl.add,
                                   reduce_res=psums[:, t:t + 1], bias=rmax[:, 0:1])
        etp = nl.ndarray((128, nkt, KT), dtype=BF16, buffer=nl.sbuf)
        for t in range(nkt):
            k0 = t * KT
            nisa.dma_transpose(dst=etp.ap([[nkt * KT, 128], [1, 1], [128, KT // 128], [1, 128]], offset=t * KT),
                               src=e_sb.ap([[S, 128], [1, 1], [128, KT // 128], [1, 128]], offset=k0))
        mm2 = nl.ndarray((128, W), dtype=F32, buffer=nl.sbuf)
        for L in range(nlt):
            po = nl.ndarray((128, W), dtype=F32, buffer=nl.psum)
            for jj in range(LT // KT):
                t = L * (LT // KT) + jj
                if t < nkt:
                    for i in range(KT // 128):
                        vt = t * (KT // 128) + i
                        nisa.nc_matmul(dst=po[:, 0:W], stationary=v_sb[:, vt, :],
                                       moving=etp[:, t, i * 128:(i + 1) * 128])
            if L == 0:
                nisa.tensor_copy(dst=mm2[:, 0:W], src=po[:, 0:W])
            else:
                nisa.tensor_tensor(dst=mm2[:, 0:W], data1=mm2[:, 0:W], data2=po[:, 0:W], op=nl.add)
        ssum = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ssum, op=nl.add, data=psums, axis=1)
        rc = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rc, data=ssum)
        pr = nl.ndarray((128, W), dtype=F32, buffer=nl.psum)
        nisa.nc_transpose(dst=pr.ap(pattern=[[W, 128], [1, W]], offset=0), data=rc.ap(pattern=[[1, 128], [0, 128]], offset=0))
        rT = nl.ndarray((128, W), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rT[:, 0:W], src=pr[:, 0:W])
        fin = nl.ndarray((128, W), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=fin[:, 0:W], data1=mm2[:, 0:W], data2=rT[:, 0:W], op=nl.multiply)
        # only the last window column (query pos) is used by the tail: att[h, :, pos] (DMA rounds fp32 -> bf16)
        nisa.dma_copy(dst=att.ap(pattern=[[S, 128], [1, 1]], offset=h * 128 * S, scalar_offset=preg, indirect_dim=2),
                      src=fin[:, W - 1:W])


def _ns_psx_exch(xa, pA, h1, xn2, g2f, hs, S, scaled=False):
    """Partial-sum exchange -> rstd; then the h1 exchange and the other half of xn2 (NS_PSX)."""
    pid = nl.program_id(0)
    o = 1 - pid
    C = S
    nisa.core_barrier(data=xa, cores=(0, 1))
    pB = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pB, src=xa.ap(pattern=[[0, 128], [1, C]], offset=o * 128 * HT * S),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
    tot = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot, data1=pA, data2=pB, op=nl.add)    # P0 + P1 == P1 + P0: same on both cores
    t = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=tot, op0=nl.multiply, operand0=1.0 / H, op1=nl.add, operand1=EPS)
    rstd = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=t, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    if not NS_POSTR:
        for kt in range(16 * pid, 16 * pid + 16):
            nisa.scalar_tensor_tensor(dst=xn2[:, kt, :], data=h1[:, kt, :], op0=nl.multiply,
                                      operand0=g2f[:, kt:kt + 1], op1=nl.multiply, operand1=rstd)
    nisa.core_barrier(data=hs, cores=(0, 1))
    if scaled:   # hs holds bf16(h1 * g2) of the other half
        xdma(dst=xn2[:, 16 * o:16 * o + 16, :], src=hs[:, 16 * o:16 * o + 16, 0:C], q=NS_XQH)
        return rstd
    xdma(dst=h1[:, 16 * o:16 * o + 16, :], src=hs[:, 16 * o:16 * o + 16, 0:C], q=NS_XQH)
    if NS_POSTR:
        _scale_g(xn2, h1, g2f, 16 * o, 16 * o + 16)
    else:
        for kt in range(16 * o, 16 * o + 16):
            nisa.scalar_tensor_tensor(dst=xn2[:, kt, :], data=h1[:, kt, :], op0=nl.multiply,
                                      operand0=g2f[:, kt:kt + 1], op1=nl.multiply, operand1=rstd)
    return rstd


# ===================================================================================================================
# Round 011: last prefill layer + final norm + LM head + argmax in one NEFF (pf_lasth_S). S <= 256: phase A computes
# only K / V for all tokens (own kv heads, NxD rounding of the norm), Q and attention only for the 128-query window
# ending at pos (_last_qattn, as S >= 1024), then the one-token tail and the decode head (head_sb). S >= 1024: the
# pf_last_tok body, then the head. The tail's two output halves meet in column 1 of the hs scratch.

def _ns_last_kv(x, cosP, sinP, cst, g1f, ones_b, wq_c, kc, vc, kq, vq, S):
    pid = nl.program_id(0)
    C = S
    xc = nl.ndarray((128, HT, C), dtype=BF16, buffer=nl.sbuf)
    for a in range(4):
        wdma(xc[:, a * 8:(a + 1) * 8, :], x[:, a * 8:(a + 1) * 8, 0:C])
    xn = _norm_fm_split(xc, g1f, ones_b, C)
    cs = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs, src=cosP[:, 0:C])
    sn = nl.ndarray((128, C), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sn, src=sinP[:, 0:C])
    ks = []
    vs = []
    for n in range(16, QKV_NT):
        wsl = _load_slab(wq_c, n * HT, HT)
        ps = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
        _gemm_tile(ps, wsl, xn, HT)
        yb = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=yb, src=ps)
        if n < 20:
            pr = nl.ndarray((128, C), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(dst=pr, stationary=cst[:, 0:128], moving=yb)
            a_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=a_, data1=yb, data2=cs, op=nl.multiply)
            b_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=b_, data1=pr, data2=sn, op=nl.multiply)
            r_ = nl.ndarray((128, C), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r_, data1=a_, data2=b_, op=nl.add)
            nisa.dma_copy(dst=kc[4 * pid + n - 16, :, 0:C], src=r_)
            if kq is not None:
                nisa.dma_copy(dst=kq[4 * pid + n - 16, :, 0:C], src=r_)
            ks.append(r_)
        else:
            G = 4 * pid + n - 20
            vl = []
            for b0 in range(0, C, 128):
                w = min(128, C - b0)
                pt = nl.ndarray((128, 128), dtype=F32, buffer=nl.psum)
                nisa.nc_matmul(dst=pt[0:w, :], stationary=yb[:, b0:b0 + w], moving=cst[:, 128:256])
                vt = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=vt[0:w, :], src=pt[0:w, :])
                nisa.dma_copy(dst=vc[b0:b0 + w, G * 128:(G + 1) * 128], src=vt[0:w, :])
                if vq is not None:
                    nisa.dma_copy(dst=vq[G, b0:b0 + w, :], src=vt[0:w, :])
                vl.append(vt)
            vs.append(vl)
    return ks, vs


LH_TOKATT_TOK = 0   # the same for pf_lasth_tok (S >= 1024): measured neutral (1024 / 4096 e2e 0.9999 / 1.0000)
LH_TOKATT = 1   # pf_lasth_ns: decode-style attention of the last token only (K/V kept in SBUF) instead of the
                # 128-query window (_last_qattn)


def _ns_last_tokattn(x, cosP, sinP, cst, g1, wq_c, ks, vs, att, S, pos, kq=None, vq=None):
    """Q (own 16 heads) and attention for the token at pos only: x[:, :, pos] -> RMSNorm -> q GEMV -> RoPE ->
    bf16(q / sqrt(d)) (as the prefill layers) -> scores over keys 0..S-1 of the core's 4 kv heads (keys > pos masked)
    -> softmax (fp32) -> P (bf16, normalized) -> P V; stores att[16 pid + h, :, pos]."""
    pid = nl.program_id(0)
    p_sb, preg = decode._load_pos(pos)
    ones_f = nl.ndarray((128, 128), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    g_sb = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_sb, src=g1)
    xcol = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xcol, src=x.ap(pattern=[[HT * S, 128], [S, HT]], offset=0, scalar_offset=preg, indirect_dim=2))
    xq = rmsnorm_col(xcol, g_sb, ones_f)
    ps_q = nl.ndarray((128, 16), dtype=F32, buffer=nl.psum)
    gemv_cols(wq_c[:, 0:16 * HT * 128], xq, 16, HT, ps_q)
    qb = nl.ndarray((128, 16), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qb, src=ps_q)
    TM = cosP.shape[1]
    cs = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs, src=cosP.ap(pattern=[[TM, 128], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
    sn = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sn, src=sinP.ap(pattern=[[TM, 128], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
    pr = nl.ndarray((128, 16), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(dst=pr, stationary=cst[:, 0:128], moving=qb)
    a_ = nl.ndarray((128, 16), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a_, data=qb, op0=nl.multiply, operand0=cs)
    b_ = nl.ndarray((128, 16), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=b_, data=pr, op0=nl.multiply, operand0=sn)
    r_ = nl.ndarray((128, 16), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=r_, data1=a_, data2=b_, op=nl.add)
    q_ = nl.ndarray((128, 16), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=q_, data=r_, op0=nl.multiply, operand0=QSCALE)
    qm = []
    for g in range(4):
        t = nl.ndarray((128, 16), dtype=BF16, buffer=nl.sbuf)
        nisa.memset(dst=t, value=0.0)
        nisa.tensor_copy(dst=t[:, 4 * g:4 * g + 4], src=q_[:, 4 * g:4 * g + 4])
        qm.append(t)
    posf_i = nl.ndarray((16, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=posf_i, src=pos.ap(pattern=[[0, 16], [1, 1]], offset=0))
    posf = nl.ndarray((16, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=posf, src=posf_i)
    iot = nl.ndarray((16, S), dtype=F32, buffer=nl.sbuf)
    nisa.iota(dst=iot, pattern=[[1, S]], offset=0, channel_multiplier=0)
    mb = nl.ndarray((16, S), dtype=F32, buffer=nl.sbuf)   # -30000 where key > pos
    nisa.tensor_scalar(dst=mb, data=iot, op0=nl.greater, operand0=posf, op1=nl.multiply, operand1=-30000.0)
    if ks is None:   # S >= 1024 (pf_lasth_tok): this core's 4 kv heads, all keys, from the exchanged kq / vq
        ks = []
        vs = []
        for g in range(4):
            kt_ = nl.ndarray((128, S), dtype=BF16, buffer=nl.sbuf)
            nisa.dma_copy(dst=kt_, src=kq[4 * pid + g, :, 0:S], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            ks.append(kt_)
            vt_ = nl.ndarray((128, S // 128, 128), dtype=BF16, buffer=nl.sbuf)
            nisa.dma_copy(dst=vt_, src=vq.ap(pattern=[[128, 128], [128 * 128, S // 128], [1, 128]],
                                             offset=(4 * pid + g) * S * 128),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            vl = []
            for i in range(S // 128):
                vl.append(vt_[:, i, :])
            vs.append(vl)
    sc = nl.ndarray((16, S), dtype=F32, buffer=nl.sbuf)
    for c0 in range(0, S, 512):
        cw = min(512, S - c0)
        ps = nl.ndarray((16, 512), dtype=F32, buffer=nl.psum)
        for g in range(4):
            nisa.nc_matmul(dst=ps[:, 0:cw], stationary=qm[g], moving=ks[g][:, c0:c0 + cw])
        nisa.tensor_tensor(dst=sc[:, c0:c0 + cw], data1=ps[:, 0:cw], data2=mb[:, c0:c0 + cw], op=nl.add)
    mx = nl.ndarray((16, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.maximum, data=sc, axis=(1,))
    nmx = nl.ndarray((16, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0)
    e = nl.ndarray((16, S), dtype=F32, buffer=nl.sbuf)
    ssum = nl.ndarray((16, 1), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=e, op=nl.exp, data=sc, bias=nmx, scale=1.0, reduce_op=nl.add, reduce_res=ssum,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rinv = nl.ndarray((16, 1), dtype=F32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rinv, data=ssum)
    pb = nl.ndarray((16, S), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pb, data=e, op0=nl.multiply, operand0=rinv)
    NTT = S // 128
    pT = nl.ndarray((128, NTT, 16), dtype=BF16, buffer=nl.sbuf)
    for i0 in range(0, NTT, 32):
        n_ = min(32, NTT - i0)
        pst = nl.ndarray((128, 32, 16), dtype=F32, buffer=nl.psum)
        for i in range(n_):
            nisa.nc_matmul(dst=pst[:, i, :], stationary=pb[:, (i0 + i) * 128:(i0 + i + 1) * 128],
                           moving=cst[0:16, 128:144])
        nisa.tensor_copy(dst=pT[:, i0:i0 + n_, :], src=pst[:, 0:n_, :])
    attn = nl.ndarray((128, 16), dtype=BF16, buffer=nl.sbuf)
    for g in range(4):
        ps_o = nl.ndarray((128, 4), dtype=F32, buffer=nl.psum)
        for i in range(NTT):
            nisa.nc_matmul(dst=ps_o, stationary=vs[g][i], moving=pT[:, i, 4 * g:4 * g + 4])
        nisa.tensor_copy(dst=attn[:, 4 * g:4 * g + 4], src=ps_o)
    nisa.dma_copy(dst=att.ap(pattern=[[S, 128], [128 * S, 16]], offset=16 * pid * 128 * S, scalar_offset=preg,
                             indirect_dim=2), src=attn)


def _lasth_head(hs, pos, toks, gf, lm, idb):
    """The last token's hidden state (both cores' halves in column 1 of hs) -> decode head (final norm, LM head,
    argmax, toks[pos+1]); returns pos+1."""
    nisa.core_barrier(data=hs, cores=(0, 1))
    h_bf = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=h_bf, src=hs.ap(pattern=[[HT * hs.shape[2], 128], [hs.shape[2], HT]], offset=1))
    p_sb, preg = decode._load_pos(pos)
    ones_f = nl.ndarray((128, 128), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    return decode.head_sb(h_bf, p_sb, preg, ones_f, toks, gf, lm, idb)


@nki.jit
def pf_lasth_ns(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, hs, ms, pos, toks, gf, lm,
                idb, S=256):
    """Last layer (S = 256: K/V for all tokens, Q / attention for the last token) + head."""
    return _lasth_ns_body(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, hs, ms, pos, toks,
                          gf, lm, idb, S)


def _lasth_ns_body(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, hs, ms, pos, toks, gf, lm,
                   idb, S):
    pid = nl.program_id(0)
    cst = _consts(consts)
    ones_b = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
    nisa.memset(dst=ones_b, value=1.0)
    g1b = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=g1b, src=g1)
    g1f = nl.ndarray((128, HT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g1f, src=g1b)
    g2b = nl.ndarray((128, HT), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=g2b, src=g2)
    g2f = nl.ndarray((128, HT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g2f, src=g2b)
    if LH_TOKATT:
        ks, vs = _ns_last_kv(x, cosP, sinP, cst, g1f, ones_b, wqkv[pid, :, :], kc, vc, None, None, S)
        _ns_last_tokattn(x, cosP, sinP, cst, g1, wqkv[pid, :, :], ks, vs, att, S, pos)
    else:
        _ns_last_kv(x, cosP, sinP, cst, g1f, ones_b, wqkv[pid, :, :], kc, vc, kq, vq, S)
        nisa.core_barrier(data=kq, cores=(0, 1))
        nisa.core_barrier(data=vq, cores=(0, 1))
        _last_qattn(x, cosP, sinP, cst, g1f, ones_b, wqkv, kq, vq, att, S, pos)
    _pf_tail(x, wo, g2f, wgu, wd, att, hs, ms, ones_b, S, pos, barrier=True, hcol1=True)
    return _lasth_head(hs, pos, toks, gf, lm, idb)


@nki.jit
def pf_lasth_tok(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, hs, ms, pos, toks, gf, lm,
                 idb, S=1024):
    """pf_last_tok + head."""
    pf_layer_tok_body(x, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, S, hs=hs, ms=ms,
                      pos=pos, hcol1=True)
    return _lasth_head(hs, pos, toks, gf, lm, idb)


def _embed_full(toks, embed, cst, S):
    """x [128, HT, S] feature-major in SBUF = embed[toks[0, s]] for all s < S (each core gathers all tokens; the
    transposes by identity matmuls are exact, as pf_embed)."""
    xs = nl.ndarray((128, HT, S), dtype=BF16, buffer=nl.sbuf)
    for b0 in range(0, S, 128):
        idx = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=toks.ap(pattern=[[1, 128], [1, 1]], offset=b0))
        rows = nl.ndarray((128, H), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=embed.ap(pattern=[[H, 128], [1, H]], offset=0, vector_offset=idx, indirect_dim=0))
        for k0 in range(0, HT, 4):
            pt = nl.ndarray((128, 4, 128), dtype=F32, buffer=nl.psum)
            for j in range(4):
                kt = k0 + j
                nisa.nc_matmul(dst=pt[:, j, :], stationary=rows[:, kt * 128:(kt + 1) * 128], moving=cst[:, 128:256])
            if (k0 // 4) % 2 == 0:
                nisa.tensor_copy(dst=xs[:, k0:k0 + 4, b0:b0 + 128], src=pt)
            else:
                nisa.activation(dst=xs[:, k0:k0 + 4, b0:b0 + 128], op=nl.copy, data=pt)
    return xs


@nki.jit
def pf_first_ns(toks, embed, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, hs, ms, xa,
                S=256):
    """Layer 0 for S <= 256 with the embedding gather fused (replaces pf_embed + pf_layer_ns for layer 0)."""
    cst = _consts(consts)
    xs = _embed_full(toks, embed, cst, S)
    return pf_layer_body(None, cosP, sinP, consts, g1, wqkv, wo, g2, wgu, wd, kc, vc, qs, att, kq, vq, xa, None, hs, S,
                         S, NS=1, ms=ms, xsb=xs)


PF_MULTI_OWN = 1   # pf_multi: the next layer starts on its own half of x from SBUF (MULTI_PRE)


def _pf_multi_body(x, t0, t1, cosP, sinP, consts, qs, att, kq, vq, hs, ms, xa, layers, S):
    """Consecutive S <= 256 layers in one NEFF (no per-layer execution preamble): layer j reads src, writes its
    output half into t0 / t1 (alternating) and both cores meet at a core barrier; the last layer's output is
    returned. Same instructions per layer as pf_layer_ns."""
    src = x
    n = len(layers)
    out = None
    xown = None
    for j in range(n):
        lw = layers[j]
        dst = None
        if j < n - 1:
            dst = t0
            if j % 2 == 1:
                dst = t1
        out = pf_layer_body(src, cosP, sinP, consts, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7], qs, att,
                            kq, vq, xa, None, hs, S, S, NS=1, ms=ms, xdst=dst, xown=xown)
        if j < n - 1:
            xown = None
            if PF_MULTI_OWN:
                xown = out[1]
            nisa.core_barrier(data=dst, cores=(0, 1))
            src = dst
    return out


@nki.jit
def pf_multi15_ns(x, t0, t1, cosP, sinP, consts, qs, att, kq, vq, hs, ms, xa,
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
                   S=256):
    """15 consecutive S <= 256 layers (pf_layer_ns) in one NEFF."""
    return _pf_multi_body(x, t0, t1, cosP, sinP, consts, qs, att, kq, vq, hs, ms, xa, [
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
    ], S)


def _pf_multi_pipe_body(x, t0, t1, cosP, sinP, consts, scr, layers, S, tok=False):
    """Consecutive pf_layer_pipe layers (S = 1024) in one NEFF. Token split: each core's next-layer input is its own
    tokens' output (written and read by the same core), so layers follow without a barrier; the q / K / V /
    attention scratch alternates between two sets (scr), so a core starting layer j+1 never overwrites what the other
    core may still read in layer j (layer j+1's K/V barrier separates layer j from layer j+2)."""
    src = x
    n = len(layers)
    out = None
    for j in range(n):
        lw = layers[j]
        dst = None
        if j < n - 1:
            dst = t0
            if j % 2 == 1:
                dst = t1
        sc = scr[j % 2]
        if tok:
            out = pf_layer_tok_body(src, cosP, sinP, consts, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7],
                                    sc[0], sc[1], sc[2], sc[3], S, xdst=dst)
        else:
            out = pf_layer_pipe_body(src, cosP, sinP, consts, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7],
                                     sc[0], sc[1], sc[2], sc[3], S, xdst=dst)
        src = dst
    return out


@nki.jit
def pf_multi15_pipe(x, t0, t1, cosP, sinP, consts, qs0, att0, kq0, vq0, qs1, att1, kq1, vq1,
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
                     S=1024):
    """15 consecutive S = 1024 layers (pf_layer_pipe) in one NEFF."""
    return _pf_multi_pipe_body(x, t0, t1, cosP, sinP, consts, [(qs0, att0, kq0, vq0), (qs1, att1, kq1, vq1)], [
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
    ], S)


@nki.jit
def pf_multi6_tok(x, t0, t1, cosP, sinP, consts, qs0, att0, kq0, vq0, qs1, att1, kq1, vq1,
                   g1_0, wqkv_0, wo_0, g2_0, wgu_0, wd_0, kc_0, vc_0,
                   g1_1, wqkv_1, wo_1, g2_1, wgu_1, wd_1, kc_1, vc_1,
                   g1_2, wqkv_2, wo_2, g2_2, wgu_2, wd_2, kc_2, vc_2,
                   g1_3, wqkv_3, wo_3, g2_3, wgu_3, wd_3, kc_3, vc_3,
                   g1_4, wqkv_4, wo_4, g2_4, wgu_4, wd_4, kc_4, vc_4,
                   g1_5, wqkv_5, wo_5, g2_5, wgu_5, wd_5, kc_5, vc_5,
                   S=4096):
    """6 consecutive S >= 2048 layers (pf_layer_tok) in one NEFF (same scheme as pf_multi15_pipe)."""
    return _pf_multi_pipe_body(x, t0, t1, cosP, sinP, consts, [(qs0, att0, kq0, vq0), (qs1, att1, kq1, vq1)], [
        (g1_0, wqkv_0, wo_0, g2_0, wgu_0, wd_0, kc_0, vc_0),
        (g1_1, wqkv_1, wo_1, g2_1, wgu_1, wd_1, kc_1, vc_1),
        (g1_2, wqkv_2, wo_2, g2_2, wgu_2, wd_2, kc_2, vc_2),
        (g1_3, wqkv_3, wo_3, g2_3, wgu_3, wd_3, kc_3, vc_3),
        (g1_4, wqkv_4, wo_4, g2_4, wgu_4, wd_4, kc_4, vc_4),
        (g1_5, wqkv_5, wo_5, g2_5, wgu_5, wd_5, kc_5, vc_5),
    ], S, True)


def _embed_chunks(toks, embed, cst, xdst, chunks, C):
    """xdst[:, :, t0:t0+C] = embed[toks[t0 .. t0+C-1]] (feature-major) for the given chunks (as pf_embed)."""
    for ch in chunks:
        for b0 in range(0, C, 128):
            t0 = ch * C + b0
            idx = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=idx, src=toks.ap(pattern=[[1, 128], [1, 1]], offset=t0))
            rows = nl.ndarray((128, H), dtype=BF16, buffer=nl.sbuf)
            nisa.dma_copy(dst=rows, src=embed.ap(pattern=[[H, 128], [1, H]], offset=0, vector_offset=idx,
                                                 indirect_dim=0))
            xs = nl.ndarray((128, HT, 128), dtype=BF16, buffer=nl.sbuf)
            for k0 in range(0, HT, 4):
                pt = nl.ndarray((128, 4, 128), dtype=F32, buffer=nl.psum)
                for j in range(4):
                    kt = k0 + j
                    nisa.nc_matmul(dst=pt[:, j, :], stationary=rows[:, kt * 128:(kt + 1) * 128], moving=cst[:, 128:256])
                if (k0 // 4) % 2 == 0:
                    nisa.tensor_copy(dst=xs[:, k0:k0 + 4, :], src=pt)
                else:
                    nisa.activation(dst=xs[:, k0:k0 + 4, :], op=nl.copy, data=pt)
            nisa.dma_copy(dst=xdst[:, :, t0:t0 + 128], src=xs)


def _pf_multi_ns_eh(x, t0, t1, cosP, sinP, consts, qs, att, kq, vq, hs, ms, xa, layers, S, emb=None, head=None):
    """pf_multi for S <= 256 with the embedding gather before layer 0 (emb = (toks, embed)) and / or the last layer
    + head after the layers (head = (last layer weights, pos, toks, gf, lm, idb)): the whole prefill in 2 NEFFs."""
    src = x
    n = len(layers)
    out = None
    xown = None
    xs = None
    if emb is not None:
        xs = _embed_full(emb[0], emb[1], _consts(consts), S)
    for j in range(n):
        lw = layers[j]
        dst = None
        if j < n - 1 or head is not None:
            dst = t0
            if j % 2 == 1:
                dst = t1
        xsb = None
        if j == 0:
            xsb = xs
        out = pf_layer_body(src, cosP, sinP, consts, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7], qs, att,
                            kq, vq, xa, None, hs, S, S, NS=1, ms=ms, xdst=dst, xown=xown, xsb=xsb)
        if dst is not None:
            xown = None
            if PF_MULTI_OWN:
                xown = out[1]
            nisa.core_barrier(data=dst, cores=(0, 1))
            src = dst
    if head is None:
        return out
    lw = head[0]
    return _lasth_ns_body(src, cosP, sinP, consts, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7], qs, att,
                          kq, vq, hs, ms, head[1], head[2], head[3], head[4], head[5], S)


def _pf_multi_pipe_eh(x, t0, t1, cosP, sinP, consts, scr, hs, ms, layers, S, emb=None, head=None):
    """The same for the S = 1024 pipe layers (embedding of the own chunks into t1 first; the last layer is
    pf_lasth_tok's body)."""
    pid = nl.program_id(0)
    src = x
    n = len(layers)
    out = None
    if emb is not None:
        _embed_chunks(emb[0], emb[1], _consts(consts), t1, _own_chunks(pid, S // PIPE_C), PIPE_C)
        src = t1
    for j in range(n):
        lw = layers[j]
        dst = None
        if j < n - 1 or head is not None:
            dst = t0
            if j % 2 == 1:
                dst = t1
        sc = scr[j % 2]
        out = pf_layer_pipe_body(src, cosP, sinP, consts, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7],
                                 sc[0], sc[1], sc[2], sc[3], S, xdst=dst)
        src = dst
    if head is None:
        return out
    nisa.core_barrier(data=src, cores=(0, 1))   # the tail reads the last token's column (either core's)
    lw = head[0]
    sc = scr[n % 2]
    pf_layer_tok_body(src, cosP, sinP, consts, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7], sc[0], sc[1],
                      sc[2], sc[3], S, hs=hs, ms=ms, pos=head[1], hcol1=True)
    return _lasth_head(hs, head[1], head[2], head[3], head[4], head[5])


@nki.jit
def pf_multi16e_ns(toks, embed, t0, t1, cosP, sinP, consts, qs, att, kq, vq, hs, ms, xa,
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
                    S=256):
    """Embedding + layers 0-15 (S <= 256) in one NEFF."""
    return _pf_multi_ns_eh(None, t0, t1, cosP, sinP, consts, qs, att, kq, vq, hs, ms, xa, [
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
    ], S, emb=(toks, embed))


@nki.jit
def pf_multi15h_ns(x, t0, t1, cosP, sinP, consts, qs, att, kq, vq, hs, ms, xa,
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
                    pos, toks, gf, lm, idb, S=256):
    """Layers 16-30 + last layer + head (S <= 256) in one NEFF (layer 15 of the argument list is model layer 31)."""
    return _pf_multi_ns_eh(x, t0, t1, cosP, sinP, consts, qs, att, kq, vq, hs, ms, xa, [
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
    ], S, head=((g1_15, wqkv_15, wo_15, g2_15, wgu_15, wd_15, kc_15, vc_15), pos, toks, gf, lm, idb))


@nki.jit
def pf_multi16e_pipe(toks, embed, t0, t1, cosP, sinP, consts, qs0, att0, kq0, vq0, qs1, att1, kq1, vq1,
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
                    S=1024):
    """Embedding + layers 0-15 (S = 1024 pipe layers) in one NEFF."""
    return _pf_multi_pipe_eh(None, t0, t1, cosP, sinP, consts, [(qs0, att0, kq0, vq0), (qs1, att1, kq1, vq1)], None,
                             None, [
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
    ], S, emb=(toks, embed))


@nki.jit
def pf_multi15h_pipe(x, t0, t1, cosP, sinP, consts, qs0, att0, kq0, vq0, qs1, att1, kq1, vq1, hs, ms,
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
                    pos, toks, gf, lm, idb, S=1024):
    """Layers 16-30 + last layer + head (S = 1024) in one NEFF (layer 15 of the argument list is model layer 31)."""
    return _pf_multi_pipe_eh(x, t0, t1, cosP, sinP, consts, [(qs0, att0, kq0, vq0), (qs1, att1, kq1, vq1)], hs, ms, [
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
    ], S, head=((g1_15, wqkv_15, wo_15, g2_15, wgu_15, wd_15, kc_15, vc_15), pos, toks, gf, lm, idb))
