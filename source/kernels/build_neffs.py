"""Build time only: compile every NKI kernel variant to a NEFF through NKI's documented route
(NKI_ARTIFACTS_DIR set before `import nki`, then the @nki.jit kernel called once with numpy arrays), cache them
under target/neffcache keyed by the kernel sources and the variant, and generate the assembler/C++ files that embed
them in the executable.

usage: build_neffs.py TARGET_DIR            (build all, write TARGET_DIR/gen/*)
       build_neffs.py --one NAME OUT.neff   (worker: build one variant; run by the pool with its own artifacts dir)
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_KEY_VERSION = "2"   # bump when the compile route itself (worker / compiler options) changes
TMAX = 4224
TOKS_LEN = TMAX + 128
DEC_BUCKETS = [384, 512, 768, 1024, 1152, 1536, 2048, 2560, 3072, 3584, 4096, 4224]
PF_BUCKETS = [128, 256, 512, 1024, 2048, 4096]
PF_NSPLIT_MAX = 512
PF_NS_MAX = 256
PF_PIPE_S = (1024,)     # buckets using pf_layer_pipe (attention interleaved with the MLP) instead of pf_layer_tok:
                        # per-layer profile at 1024 3.41 -> 3.33 ms; at 4096 every interleaved variant was slower
PF_LASTH = (256, 1024, 2048, 4096)   # buckets with pf_lasth_S (last layer + final norm + LM head + argmax);
                                     # S = 256 needs prompts >= 128 tokens (the 128-query window), true for that bucket
PF_FIRST = (128, 256)
PF_EH = ()      # buckets with pf_multi16e_S + pf_multi15h_S (2 executions per prefill)
PF_MULTI = {256: 15, 1024: 15}   # bucket: G (4096 with G = 6 measured 0.26% slower: not built); pf_multi<G>_S runs layers 1..30 as 30 / G executions of one
                                          # G-layer NEFF (4096: 6, a 15-layer NEFF would compile for ~40 min)   # buckets with pf_first_S (pf_embed + layer 0 in one NEFF)
GRP_BUCKETS = DEC_BUCKETS
GRP_G = 16              # decode layers per NEFF (dec_grp16 in decode.py)
GRP_FUSE = 1            # dec_grp16e (embed + layers 0-15) and dec_grp16h (layers 16-31 + head): 2 executions per step


def variants():
    v = {"dec_embed": ("decode", "dec_embed", {}), "dec_head": ("decode", "dec_head", {})}
    for b in DEC_BUCKETS:
        v[f"dec_layer_{b}"] = ("decode", "dec_layer", {"B": b})
    for b in GRP_BUCKETS:
        if not GRP_FUSE:
            v[f"dec_grp{GRP_G}_{b}"] = ("decode", f"dec_grp{GRP_G}", {"B": b})
        else:   # first group with dec_embed fused, second with dec_head fused
            v[f"dec_grp{GRP_G}e_{b}"] = ("decode", f"dec_grp{GRP_G}e", {"B": b})
            v[f"dec_grp{GRP_G}h_{b}"] = ("decode", f"dec_grp{GRP_G}h", {"B": b})
    for S in PF_BUCKETS:
        v[f"pf_embed_{S}"] = ("prefill", "pf_embed", {"S": S})
        if S <= PF_NS_MAX:       # heads / N-split (bf16 activation exchanges): short prompts
            v[f"pf_layer_{S}"] = ("prefill", "pf_layer_ns", {"S": S})
        elif S <= PF_NSPLIT_MAX:   # split by heads / intermediate tiles (weights read once)
            v[f"pf_layer_{S}"] = ("prefill", "pf_layer", {"S": S, "CD": 512})
        else:                    # split by tokens: long prompts (compute-bound)
            v[f"pf_layer_{S}"] = ("prefill", "pf_layer_pipe" if S in PF_PIPE_S else "pf_layer_tok", {"S": S})
        v[f"pf_pick_{S}"] = ("prefill", "pf_pick", {"S": S})
        if S <= PF_NS_MAX:       # last layer: o_proj/MLP for the last prompt token only
            v[f"pf_last_{S}"] = ("prefill", "pf_last_ns", {"S": S})
        elif S > PF_NSPLIT_MAX:
            v[f"pf_last_{S}"] = ("prefill", "pf_last_tok", {"S": S})
        if S in PF_MULTI:        # PF_MULTI[S] layers in one NEFF (round 011)
            G = PF_MULTI[S]
            kind = "ns" if S <= PF_NS_MAX else ("pipe" if S in PF_PIPE_S else "tok")
            v[f"pf_multi{G}_{S}"] = ("prefill", f"pf_multi{G}_{kind}", {"S": S})
        if S in PF_EH:           # the whole prefill in 2 NEFFs: embed + layers 0-15, layers 16-30 + last + head
            kind = "ns" if S <= PF_NS_MAX else "pipe"
            v[f"pf_multi16e_{S}"] = ("prefill", f"pf_multi16e_{kind}", {"S": S})
            v[f"pf_multi15h_{S}"] = ("prefill", f"pf_multi15h_{kind}", {"S": S})
        if S in PF_FIRST:        # layer 0 with the embedding gather fused (round 011)
            v[f"pf_first_{S}"] = ("prefill", "pf_first_ns", {"S": S})
        if S in PF_LASTH:        # last layer + head in one NEFF (round 011)
            v[f"pf_lasth_{S}"] = ("prefill", "pf_lasth_ns" if S <= PF_NS_MAX else "pf_lasth_tok", {"S": S})
    return v


def make_args(fn, kw):
    import numpy as np
    import ml_dtypes
    bf = ml_dtypes.bfloat16
    z = lambda shape, dt=bf: np.zeros(shape, dt)

    def layer_args():   # g1, wqkv, wo, g2, wgu, wd, kc, vc
        return [z((128, 32)), z((2, 128, 24 * 32 * 128)), z((2, 128, 32 * 16 * 128)), z((128, 32)),
                z((2, 128, 112 * 32 * 128)), z((2, 128, 32 * 56 * 128)), z((8, 128, TMAX)), z((TMAX, 1024))]
    i32 = np.int32
    f32 = np.float32
    pos = np.zeros((1, 1), i32)
    if fn == "dec_embed":
        return [z((1, TOKS_LEN), i32), pos, z((128256, 4096))]
    if fn == "dec_head":
        return [z((128, 32)), pos, z((1, TOKS_LEN), i32), z((128, 32)), z((2, 128, 517 * 32 * 128)), z((2, 128, 1), f32)]
    if fn == "dec_layer":
        return [z((128, 32)), pos, z((TMAX, 128), f32), z((TMAX, 128), f32), z((128, 256))] + layer_args()
    if fn == "dec_grp16e":
        return [z((1, TOKS_LEN), i32), pos, z((128256, 4096)), z((TMAX, 128), f32), z((TMAX, 128), f32),
                z((128, 256))] + layer_args() * 16
    if fn == "dec_grp16h":
        return ([z((128, 32)), pos, z((TMAX, 128), f32), z((TMAX, 128), f32), z((128, 256))] + layer_args() * 16 +
                [z((1, TOKS_LEN), i32), z((128, 32)), z((2, 128, 517 * 32 * 128)), z((2, 128, 1), f32)])
    if fn.startswith("dec_grp"):
        G = int(fn[7:])
        return [z((128, 32)), pos, z((TMAX, 128), f32), z((TMAX, 128), f32), z((128, 256))] + layer_args() * G
    S = kw.get("S", 0)
    if fn == "pf_embed":
        return [z((1, TOKS_LEN), i32), z((128256, 4096)), z((128, 256))]
    if fn == "pf_pick":
        return [z((128, 32, S)), pos]
    if fn in ("pf_layer_tok", "pf_layer_pipe"):
        return make_args("pf_layer", kw)[:16]
    if fn in ("pf_lasth_ns", "pf_lasth_tok"):
        return make_args("pf_last_ns", kw) + [z((1, TOKS_LEN), i32), z((128, 32)), z((2, 128, 517 * 32 * 128)),
                                              z((2, 128, 1), f32)]
    if fn.startswith("pf_multi16e") or fn.startswith("pf_multi15h"):
        la = make_args("pf_layer", kw)
        sx = la[0]
        pipe = fn.endswith("_pipe")
        if pipe:
            scr = la[12:16] + [z(a.shape, a.dtype) for a in la[12:16]]
        else:
            scr = [la[12], la[13], la[14], la[15], z((128, 32, S)), z((128, 112, S)), z((2, 128, 32, S), f32)]
        head = [pos, z((1, TOKS_LEN), i32), z((128, 32)), z((2, 128, 517 * 32 * 128)), z((2, 128, 1), f32)]
        if fn.startswith("pf_multi16e"):
            return [z((1, TOKS_LEN), i32), z((128256, 4096)), z(sx.shape), z(sx.shape), la[1], la[2], la[3]] + scr + \
                la[4:12] * 16
        hm = [z((128, 32, S)), z((128, 112, S))] if pipe else []
        return [sx, z(sx.shape), z(sx.shape), la[1], la[2], la[3]] + scr + hm + la[4:12] * 16 + head
    if fn.startswith("pf_multi"):
        G = int(fn[8:].split("_")[0])
        la = make_args("pf_layer", kw)
        sx = la[0]
        if fn.endswith("_pipe") or fn.endswith("_tok"):
            return [sx, z(sx.shape), z(sx.shape), la[1], la[2], la[3]] + la[12:16] + [z(a.shape, a.dtype) for a in la[12:16]] + la[4:12] * G
        return [sx, z(sx.shape), z(sx.shape), la[1], la[2], la[3], la[12], la[13], la[14], la[15],
                z((128, 32, S)), z((128, 112, S)), z((2, 128, 32, S), f32)] + la[4:12] * G
    if fn == "pf_first_ns":
        return [z((1, TOKS_LEN), i32), z((128256, 4096))] + make_args("pf_layer_ns", kw)[1:]
    if fn in ("pf_last_ns", "pf_last_tok"):
        return make_args("pf_layer", kw)[:16] + [z((128, 32, S)), z((128, 112, S)), pos]
    if fn == "pf_layer_ns":
        return make_args("pf_layer", kw)[:16] + [z((128, 32, S)), z((128, 112, S)), z((2, 128, 32, S), f32)]
    if fn == "pf_layer":
        return [z((128, 32, S)), z((128, TMAX), f32), z((128, TMAX), f32), z((128, 256)), z((128, 32)),
                z((2, 128, 24 * 32 * 128)), z((2, 128, 32 * 16 * 128)), z((128, 32)), z((2, 128, 112 * 32 * 128)),
                z((2, 128, 32 * 56 * 128)), z((8, 128, TMAX)), z((TMAX, 1024)), z((32, 128, S)), z((32, 128, S)),
                z((8, 128, S)), z((8, S, 128)),
                z((2, 128, 32, S), f32), z((2, 128, 32, S), f32), z((128, 32, S))]
    raise SystemExit("unknown kernel " + fn)


def source_hash(mod):
    """Key of a variant's sources: its kernel module and the shared helpers (the argument shapes are added per
    variant in build_all)."""
    h = hashlib.sha256()
    files = [mod + ".py", "common.py"]
    if mod == "prefill":
        files += ["pattn.py", "decode.py"]
        for d, _, fs in sorted(os.walk(os.path.join(HERE, "nkattn"))):
            files += sorted(os.path.relpath(os.path.join(d, f), HERE) for f in fs if f.endswith(".py"))
    for f in files:
        h.update(f.encode())
        h.update(open(os.path.join(HERE, f), "rb").read())
    return h.hexdigest()


def worker(name, out):
    assert os.environ.get("NKI_ARTIFACTS_DIR"), "worker needs NKI_ARTIFACTS_DIR"
    sys.path.insert(0, HERE)
    import importlib
    mod, fn, kw = variants()[name]
    m = importlib.import_module(mod)
    kern = getattr(m, fn)
    args = make_args(fn, kw)
    if fn == "pf_pick":
        kw = {}
    art = os.environ["NKI_ARTIFACTS_DIR"]
    try:
        kern[2](*args, **kw)
    except Exception as e:  # the NEFF is produced before the single run; a busy core only fails the run
        neff = os.path.join(art, "kernel.neff")
        if not os.path.exists(neff):
            raise
        print(f"[{name}] warning: compiled, but the test run failed: {e}", file=sys.stderr)
    shutil.copy(os.path.join(art, "kernel.neff"), out)


def build_all(target):
    target = os.path.abspath(target)
    cache = os.path.join(target, "neffcache")
    gen = os.path.join(target, "gen")
    work = os.path.join(target, "neffbuild")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(gen, exist_ok=True)
    vs = variants()
    paths = {}
    todo = []
    for name, (mod, fn, kw) in vs.items():
        shapes = [(a.shape, str(a.dtype)) for a in make_args(fn, kw)]
        key = hashlib.sha256((source_hash(mod) + name + fn + json.dumps(kw, sort_keys=True) + repr(shapes) +
                              BUILD_KEY_VERSION).encode()).hexdigest()[:20]
        p = os.path.join(cache, f"{name}-{key}.neff")
        paths[name] = p
        if not os.path.exists(p):
            todo.append(name)
    print(f"NEFFs: {len(vs)} variants, {len(todo)} to build", flush=True)

    def one(name):
        art = os.path.join(work, name)
        shutil.rmtree(art, ignore_errors=True)
        os.makedirs(art)
        env = {**os.environ, "NKI_ARTIFACTS_DIR": art, "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2"}
        t0 = time.time()
        tmp = paths[name] + ".tmp"
        with open(os.path.join(art, "build.log"), "w") as log:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", name, tmp], env=env,
                               stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0 or not os.path.exists(tmp):
            tail = open(os.path.join(art, "build.log")).read()[-4000:]
            raise RuntimeError(f"build of {name} failed:\n{tail}")
        os.replace(tmp, paths[name])
        shutil.rmtree(art, ignore_errors=True)
        print(f"  built {name} in {time.time() - t0:.0f} s", flush=True)

    nworkers = int(os.environ.get("NEFF_BUILD_JOBS", "6"))
    with ThreadPoolExecutor(nworkers) as ex:
        list(ex.map(one, todo))
    # embed
    names = list(vs)
    with open(os.path.join(gen, "neffs.S"), "w") as f:
        f.write(".section .rodata\n")
        for i, n in enumerate(names):
            f.write(f".balign 64\n.global neff_{i}_start\nneff_{i}_start:\n.incbin \"{paths[n]}\"\n"
                    f".global neff_{i}_end\nneff_{i}_end:\n")
        f.write(".section .note.GNU-stack,\"\",@progbits\n")
    with open(os.path.join(gen, "neff_table.cpp"), "w") as f:
        f.write("struct EmbNeff { const char* name; const unsigned char* start; const unsigned char* end; };\n")
        for i in range(len(names)):
            f.write(f"extern \"C\" const unsigned char neff_{i}_start[], neff_{i}_end[];\n")
        f.write("extern const EmbNeff g_neffs[] = {\n")
        for i, n in enumerate(names):
            f.write(f"  {{\"{n}\", neff_{i}_start, neff_{i}_end}},\n")
        f.write("};\nextern const int g_nneffs = %d;\n" % len(names))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--one":
        worker(sys.argv[2], sys.argv[3])
    else:
        build_all(sys.argv[1] if len(sys.argv) > 1 else "target")
