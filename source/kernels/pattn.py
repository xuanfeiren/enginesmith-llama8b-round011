# Derived from the NKI Library's attention_cte (github.com/aws-neuron/nki-library, Apache License 2.0,
# Copyright Amazon.com, Inc. or its affiliates): same instruction sequence, restructured into per-group stages.
"""Causal prefill attention for one (head, 128-query group) at a time, so callers can interleave its stages with
other work in program order (each engine runs its instruction stream in order).

It reproduces, instruction for instruction, what the vendored nkilib attention_cte (nkattn/, with the EngineSmith
changes) does for our configuration: one K/V section (S <= 8192), q pre-scaled (scale 1.0), causal, tp_q=False,
tp_k=False, tp_out=True, fp32 softmax and matmul outputs. Per group:
  MM1 per 512-key tile into PSUM (the diagonal tile stops at key 128*(grp+1)); diagonal tiles masked by
  range_select (+ row max), other tiles tensor_scalar_reduce(x*1.0, max); max over the tiles' maxima (negated);
  exp with the row sum per tile (activation_reduce, bias = -max), P bf16; P^T by DMA transposes; MM2 per 2048-key
  tile into PSUM ([128 d, 128 q]), summed over the 2048-key tiles in fp32; sum of the per-tile sums; reciprocal;
  reciprocal transposed with nc_transpose; O^T * 1/sum in fp32, stored (bf16) by DMA.
Same instructions, operand dtypes and reduction shapes, so the same bytes (checked against nkilib, lab/attn).
"""
import nki.language as nl
import nki.isa as nisa

BF16 = nl.bfloat16
F32 = nl.float32
FMIN = -3.4028235e38
KT = 512      # key tile (MM1, mask, exp)
LT = 2048     # large key tile (MM2 PSUM accumulation)
QG = 128      # query group
PE_TP_COPY_ENGINE = nisa.engine.vector
CHAIN_MAX = 1   # row max chained over a group's key tiles in the Vector accumulator (one read-out per group)


def attn_consts(S):
    """Per-kernel constants [zero bias column, causal upper bounds iota[p, g] = 128*g + p, S, recip scratch]."""
    zero = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=zero, value=0.0)
    ngrp = S // QG
    ubs = nl.ndarray((128, ngrp), dtype=F32, buffer=nl.sbuf)
    nisa.iota(dst=ubs, pattern=[[128, ngrp]], offset=0, channel_multiplier=1)
    return [zero, ubs, S]


def ntiles_of(grp):
    """Number of 512-key tiles with compute for query group grp."""
    return (grp * QG + QG - 1) // KT + 1


def nf_of(grp, t, S):
    """Keys computed in key tile t for group grp (the diagonal tile stops at key 128*(grp+1))."""
    return min(KT, S - t * KT, max(1, (grp + 1) * QG - t * KT))


def load_kv(kq, vq, kvh, nkeys):
    """K^T [128 d, nkeys] and V tiles [128 keys, nkeys/128, 128 d] of kv head kvh into SBUF (keys 0..nkeys-1)."""
    k_sb = nl.ndarray((128, nkeys), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_sb, src=kq[kvh, :, 0:nkeys], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    v_sb = nl.ndarray((128, nkeys // 128, 128), dtype=BF16, buffer=nl.sbuf)
    S = vq.shape[1]
    nisa.dma_copy(dst=v_sb, src=vq.ap(pattern=[[128, 128], [128 * 128, nkeys // 128], [1, 128]], offset=kvh * S * 128),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    return k_sb, v_sb


def stage_qk(ac, qs, h, grp, k_sb, q_sb=None):
    """Q load, MM1 + mask + per-tile max, negated row max. Returns the group state."""
    zero = ac[0]
    ubs = ac[1]
    S = ac[2]
    nt = ntiles_of(grp)
    nkt_all = (S + KT - 1) // KT
    if q_sb is None:
        q_sb = nl.ndarray((128, QG), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_sb, src=qs[h, :, grp * QG:(grp + 1) * QG], dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.sync)
    if CHAIN_MAX:   # the row max accumulated over the tiles in the Vector engine's accumulator, read out once
        pm1 = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    else:
        pmax = nl.ndarray((128, nkt_all), dtype=F32, buffer=nl.sbuf)
        nisa.memset(dst=pmax, value=FMIN)
    masked = nl.ndarray((128, nt * KT), dtype=F32, buffer=nl.sbuf)
    for t in range(nt):
        k0 = t * KT
        nf = nf_of(grp, t, S)
        ps = nl.ndarray((128, KT), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps[:, 0:nf], stationary=q_sb, moving=k_sb[:, k0:k0 + nf])
        if CHAIN_MAX:
            cmd = nisa.reduce_cmd.reset_reduce if t == 0 else nisa.reduce_cmd.reduce
            res = pm1 if t == nt - 1 else None
        else:
            cmd = nisa.reduce_cmd.reset_reduce
            res = pmax[:, t:t + 1]
        if grp * QG < k0 + KT:   # diagonal tile
            nisa.range_select(dst=masked[:, k0:k0 + nf], on_true_tile=ps[:, 0:nf], on_false_value=FMIN,
                              comp_op0=nl.greater_equal, comp_op1=nl.less_equal, bound0=zero[:, 0:1],
                              bound1=ubs[:, grp:grp + 1], reduce_op=nl.maximum, reduce_res=res,
                              reduce_cmd=cmd, range_start=k0)
        else:
            nisa.tensor_scalar_reduce(dst=masked[:, k0:k0 + nf], data=ps[:, 0:nf], op0=nl.multiply, operand0=1.0,
                                      reduce_op=nl.maximum, reduce_res=res, reduce_cmd=cmd)
    smax = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    if CHAIN_MAX:   # max over tiles is exact in any order; -max as the original reduce over the per-tile maxima
        nisa.tensor_reduce(dst=smax[:, 0:1], op=nl.maximum, data=pm1, axis=1, negate=True)
    else:
        nisa.tensor_reduce(dst=smax[:, 0:1], op=nl.maximum, data=pmax, axis=1, negate=True)
    rmax = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rmax, src=smax)
    return [h, grp, nt, masked, rmax]


def stage_exp(ac, st, with_tp=True, pe_tp=False):
    """exp(x - max) per key tile with its row sum (and the P^T blocks unless with_tp is False: stage_tp)."""
    grp = st[1]
    nt = st[2]
    S = ac[2]
    nkt_all = (S + KT - 1) // KT
    psums = nl.ndarray((128, nkt_all), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=psums, value=0.0)
    e_sb = nl.ndarray((128, nt * KT), dtype=BF16, buffer=nl.sbuf)
    for t in range(nt):
        k0 = t * KT
        nf = nf_of(grp, t, S)
        nisa.activation_reduce(dst=e_sb[:, k0:k0 + nf], op=nl.exp, data=st[3][:, k0:k0 + nf], reduce_op=nl.add,
                               reduce_res=psums[:, t:t + 1], bias=st[4][:, 0:1])
    st.append(psums)
    st.append(e_sb)
    st.append(None)
    if with_tp:
        stage_tp(ac, st, pe_tp)


def stage_tp(ac, st, pe_tp=False):
    """P^T blocks [key in block, tile, block*128 + q]: TensorE transposes (pe_tp) or DMA transposes (as nkilib)."""
    grp = st[1]
    nt = st[2]
    S = ac[2]
    e_sb = st[6]
    etp = nl.ndarray((128, nt, KT), dtype=BF16, buffer=nl.sbuf)
    W = nt * KT
    for t in range(nt):
        k0 = t * KT
        nf = nf_of(grp, t, S)
        nfo = nf // 128
        nfi = nf % 128
        if pe_tp:   # exact data movement; one PSUM -> SBUF copy per key tile
            pt = nl.ndarray((128, KT), dtype=BF16, buffer=nl.psum)
            for b in range(nfo):
                nisa.nc_transpose(dst=pt[:, b * 128:(b + 1) * 128], data=e_sb[:, k0 + b * 128:k0 + (b + 1) * 128],
                                  engine=nisa.engine.tensor)
            nisa.tensor_copy(dst=etp[:, t, 0:nfo * 128], src=pt[:, 0:nfo * 128], engine=PE_TP_COPY_ENGINE)
            nfo = 0
        if nfo >= 1:
            nisa.dma_transpose(dst=etp.ap([[nt * KT, 128], [1, 1], [128, nfo], [1, 128]], offset=t * KT),
                               src=e_sb.ap([[W, 128], [1, 1], [128, nfo], [1, 128]], offset=k0))
        if nfi > 0:
            nisa.dma_transpose(dst=etp.ap([[nt * KT, nfi], [1, 1], [128, 1], [1, 128]], offset=t * KT + nfo * 128),
                               src=e_sb.ap([[W, 128], [1, 1], [128, 1], [1, nfi]], offset=k0 + nfo * 128))
    st[7] = etp


def stage_pv(ac, st, v_sb):
    """MM2 per 2048-key tile, fp32 sum over those tiles."""
    grp = st[1]
    nt = st[2]
    etp = st[7]
    mm2 = nl.ndarray((128, QG), dtype=F32, buffer=nl.sbuf)
    nlt = (nt * KT + LT - 1) // LT
    for L in range(nlt):
        po = nl.ndarray((128, QG), dtype=F32, buffer=nl.psum)
        for j in range(LT // KT):
            t = L * (LT // KT) + j
            if t < nt:
                for i in range(KT // 128):
                    vt = t * (KT // 128) + i       # 128-key tile index
                    if vt * 128 <= grp * QG + QG - 1 and vt * 128 < ac[2]:
                        nisa.nc_matmul(dst=po[:, 0:QG], stationary=v_sb[:, vt, :], moving=etp[:, t, i * 128:(i + 1) * 128])
        if L == 0:
            nisa.tensor_copy(dst=mm2[:, 0:QG], src=po[:, 0:QG])
        else:
            nisa.tensor_tensor(dst=mm2[:, 0:QG], data1=mm2[:, 0:QG], data2=po[:, 0:QG], op=nl.add)
    st.append(mm2)


def stage_wb(ac, st, att, recip_all, dst_sb=None):
    """1/sum, transposed, times O^T; store bf16 O^T to att[h, :, grp*128:...] (DMA converts fp32 -> bf16).
    recip_all: [128, ngrp] fp32 scratch (the reciprocal column of group grp is written at column grp)."""
    h = st[0]
    grp = st[1]
    ngrp = ac[2] // QG
    ssum = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ssum, op=nl.add, data=st[5], axis=1)
    nisa.reciprocal(dst=recip_all[:, grp:grp + 1], data=ssum)
    pr = nl.ndarray((128, QG), dtype=F32, buffer=nl.psum)
    nisa.nc_transpose(dst=pr.ap(pattern=[[QG, 128], [1, QG]], offset=0),
                      data=recip_all.ap(pattern=[[ngrp, 128], [0, 128]], offset=grp))
    rT = nl.ndarray((128, QG), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rT[:, 0:QG], src=pr[:, 0:QG])
    if dst_sb is not None:   # bf16 O^T straight into an SBUF tile (engine rounding; checked equal to the DMA's)
        nisa.tensor_tensor(dst=dst_sb, data1=st[8][:, 0:QG], data2=rT[:, 0:QG], op=nl.multiply)
        return
    fin = nl.ndarray((128, QG), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=fin[:, 0:QG], data1=st[8][:, 0:QG], data2=rT[:, 0:QG], op=nl.multiply)
    S = att.shape[2]
    nisa.dma_copy(dst=att.ap(pattern=[[S, 128], [1, QG]], offset=h * 128 * S + grp * QG),
                  src=fin.ap(pattern=[[QG, 128], [1, QG]], offset=0))


def attention_heads(qs, kq, vq, att):
    """Whole causal attention, heads split over the two cores as nkilib's LNC2 sharding (core c: q heads 16c..16c+15,
    kv heads 4c..4c+3), groups software-pipelined as nkilib (QK of g, exp of g-1, PV + write-back of g-2)."""
    S = qs.shape[2]
    pid = nl.program_id(0)
    ac = attn_consts(S)
    recip = nl.ndarray((128, S // QG), dtype=F32, buffer=nl.sbuf)
    NG = S // QG
    for hh in range(16):
        h = pid * 16 + hh
        if hh % 4 == 0:
            k_sb, v_sb = load_kv(kq, vq, pid * 4 + hh // 4, S)
        sts = []
        for i in range(NG + 2):
            if i < NG:
                sts.append(stage_qk(ac, qs, h, i, k_sb))
            if i >= 1 and i - 1 < NG:
                stage_exp(ac, sts[i - 1])
            if i >= 2:
                stage_pv(ac, sts[i - 2], v_sb)
                stage_wb(ac, sts[i - 2], att, recip)


# ---------------------------------------------------------------------------------------------------------------
# Transposed causal attention (round 009, S >= 2048), not derived from nkilib.
# Scores are computed as S^T [128 keys, 512 queries] (K block stationary, Q^T moving), so exp(S^T) is directly the
# moving operand of PV (O^T [128 d, queries] = V^T P^T: no P transposes); the row sums come from a ones-stationary
# matmul over P^T; there is no row max: P = exp(s) without a shift. That is safe for this model (measured on CPU,
# layers 0-30 at 4096 tokens, random-token and English prompts: every row max of the scaled scores in [-14.3, +24.4],
# fp32/bf16 exp overflows at 88.7) and softmax is shift invariant. Not bitwise vs NxD: scores not rounded to bf16, no
# shift, row sums over bf16 P, P^T tiles of 2 / 4 key blocks added in bf16 before the sums matmul.
# Work unit: (head, QC-query chunk j, 128-key block kb <= NB*j + NB-1); unit pairs (kb even, kb+1) share a two-bank PSUM
# tile and one exp. Diagonal blocks are masked on P^T (GpSimd affine_select: exp of the unmasked scores is finite).
QC = 512        # queries per chunk (one PSUM bank of fp32)
TD = 3          # pipeline distance (units) between S^T (PE) and PV/sums (PE); deeper starves PSUM (measured slower)
TB = 2          # distance of the exp (ACT), applied per unit pair
SUMS_QUAD = 1   # non-diagonal key blocks: P^T of 4 blocks summed in bf16 (two levels) before one sums matmul


def _load_q(qs, h, S):
    q_sb = nl.ndarray((128, S), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=qs[h, :, :], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    return q_sb


def attention_t(qs, kq, vq, att):
    """Same contract as attention_heads (heads 16c.. on core c), S a multiple of 512."""
    S = qs.shape[2]
    pid = nl.program_id(0)
    ones = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    NQC = S // QC
    NB = QC // 128       # 128-key blocks per query chunk
    units = []           # [hh, j, kb, q0]: q0 = first query of the chunk that sees key block kb
    for hh in range(16):
        for j in range(NQC):
            for kb in range(NB * j + NB):
                units.append([hh, j, kb, max(0, kb - NB * j) * 128])
    NU = len(units)
    kvs = [None, None]
    qsb = [None]
    pss = [None] * NU     # [psum pair tile, V tiles]
    pts = [None] * NU     # [P^T pair tile, half offset, V tiles]
    acc = [None, None]    # PSUM O^T and row sums of the current chunk
    p2q = [None]
    for u in range(NU + TD):
        if u < NU:   # stage A: S^T of unit u (PE) into half u % 2 of a two-bank PSUM tile
            hh = units[u][0]
            j = units[u][1]
            kb = units[u][2]
            q0 = units[u][3]
            h = pid * 16 + hh
            if j == 0 and kb == 0:
                if hh % 4 == 0:
                    k_sb, v_sb = load_kv(kq, vq, pid * 4 + hh // 4, S)
                    kvs[0] = k_sb
                    kvs[1] = v_sb
                qsb[0] = _load_q(qs, h, S)
            if u % 2 == 0:
                ps = nl.ndarray((128, 2 * QC), dtype=F32, buffer=nl.psum)
            else:
                ps = pss[u - 1][0]
            o = (u % 2) * QC
            nisa.nc_matmul(dst=ps[:, o + q0:o + QC], stationary=kvs[0][:, kb * 128:(kb + 1) * 128],
                           moving=qsb[0][:, j * QC + q0:(j + 1) * QC])
            pss[u] = [ps, kvs[1]]
        b = u - TB
        if b >= 1 and b % 2 == 1 and b < NU:   # stage B: P^T = exp(S^T) of the unit pair (b-1, b) (ACT)
            ps = pss[b][0]
            qa = units[b - 1][3]
            qb = units[b][3]
            pt = nl.ndarray((128, 2 * QC), dtype=BF16, buffer=nl.sbuf)
            if qb == 0:   # one instruction over both banks
                nisa.activation(dst=pt[:, 0:2 * QC], op=nl.exp, data=ps[:, 0:2 * QC])
            else:         # diagonal pair: only the written ranges (an exp over unwritten PSUM raced, measured)
                nisa.activation(dst=pt[:, qa:QC], op=nl.exp, data=ps[:, qa:QC])
                nisa.activation(dst=pt[:, QC + qb:2 * QC], op=nl.exp, data=ps[:, QC + qb:2 * QC])
                for side in range(2):   # P^T[key p, query c] = 0 where p > c on the diagonal 128 x 128 blocks
                    oo = side * QC + (qb if side else qa)
                    nisa.affine_select(dst=pt[:, oo:oo + 128], pattern=[[1, 128]], channel_multiplier=-1,
                                       on_true_tile=pt[:, oo:oo + 128], on_false_value=0.0,
                                       cmp_op=nl.greater_equal)
            pts[b - 1] = [pt, 0, pss[b - 1][1]]
            pts[b] = [pt, QC, pss[b][1]]
            pss[b - 1] = None
            pss[b] = None
        if u >= TD:   # stage C: PV (+ row sums per pair / quad) of unit w (PE); chunk epilogue (DVE)
            w = u - TD
            hh = units[w][0]
            j = units[w][1]
            kb = units[w][2]
            q0 = units[w][3]
            pt = pts[w][0]
            o = pts[w][1]
            v_sb = pts[w][2]
            if kb == 0:
                acc[0] = nl.ndarray((128, QC), dtype=F32, buffer=nl.psum)
                acc[1] = nl.ndarray((128, QC), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(dst=acc[0][:, q0:QC], stationary=v_sb[:, kb, :], moving=pt[:, o + q0:o + QC])
            if kb % 2 == 1:   # bf16(P_a + P_b) on the Vector engine, then one sums matmul per pair (or quad)
                qa = units[w - 1][3]
                p2 = nl.ndarray((128, QC), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=p2[:, q0:QC], data1=pt[:, q0:QC], data2=pt[:, QC + q0:2 * QC], op=nl.add)
                quad = SUMS_QUAD and kb - kb % 4 + 3 < NB * j   # non-diagonal quad (kb-3 .. kb): one more bf16 add
                if quad and kb % 4 == 1:
                    p2q[0] = p2
                else:
                    if quad:
                        p4 = nl.ndarray((128, QC), dtype=BF16, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=p4, data1=p2q[0], data2=p2, op=nl.add)
                        p2 = p4
                    if qa < q0:   # diagonal pair: queries only the first block sees
                        nisa.nc_matmul(dst=acc[1][:, qa:q0], stationary=ones, moving=pt[:, qa:q0])
                    nisa.nc_matmul(dst=acc[1][:, q0:QC], stationary=ones, moving=p2[:, q0:QC])
                pts[w - 1] = None
                pts[w] = None
            if kb == NB * j + NB - 1:   # O^T * (1 / sum), bf16, stored
                # both PSUM accumulators copied out first (frees the banks for the next chunk: 1.64 -> 1.52 ms
                # alone), then 1/sum (3.4 us per [128, 512]) and the multiply from SBUF
                oc = nl.ndarray((128, QC), dtype=F32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=oc, src=acc[0], engine=nisa.engine.vector)
                zc = nl.ndarray((128, QC), dtype=F32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=zc, src=acc[1], engine=nisa.engine.vector)
                rz = nl.ndarray((128, QC), dtype=F32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rz, data=zc)
                ob = nl.ndarray((128, QC), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=ob, data1=oc, data2=rz, op=nl.multiply)
                nisa.dma_copy(dst=att[pid * 16 + hh, :, j * QC:(j + 1) * QC], src=ob)
