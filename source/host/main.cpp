// Llama-3.1-8B-Instruct inference engine for one logical NeuronCore (Trainium2, LNC=2).
// Host: loads weights (re-laid out for the NKI kernels, values unchanged), schedules embedded NEFFs through libnrt,
// speaks the EngineSmith JSON-lines protocol. The model is computed entirely by the NEFFs.
#include <nrt/nrt.h>
#include <nrt/nrt_async.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_profile.h>

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "json.h"

// ------------------------------------------------------------------ model constants
static const int H = 4096, HT = 32, NL = 32, IT = 112, ITC = 56, VOCAB = 128256, VTC = 501;
// LM head vocab split: core 0 takes VTC - HSK tiles, core 1 VTC + HSK (core 0 streams ~7% slower; ids unchanged)
static const int HSK = 16, VTCMAX = VTC + HSK;
static const int TMAX = 4224, TOKS_LEN = TMAX + 128;
static const int QKV_NT = 24, GU_G = 8;

// ------------------------------------------------------------------ embedded NEFFs (generated neff_table.cpp)
struct EmbNeff {
  const char* name;
  const unsigned char* start;
  const unsigned char* end;
};
extern const EmbNeff g_neffs[];
extern const int g_nneffs;

// ------------------------------------------------------------------ utilities
using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
static bool g_verbose = false;
static bool g_timing = getenv("ENGINE_TIMING") != nullptr;   // diagnostics: host-side timing of a prefill
static void logf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fflush(stderr);
}
#define CK(x)                                                                                              \
  do {                                                                                                     \
    NRT_STATUS s_ = (x);                                                                                   \
    if (s_ != NRT_SUCCESS) {                                                                               \
      char b_[512];                                                                                        \
      snprintf(b_, sizeof b_, "%s:%d %s -> %d (%s)", __FILE__, __LINE__, #x, (int)s_, nrt_get_status_as_str(s_)); \
      throw std::runtime_error(b_);                                                                        \
    }                                                                                                      \
  } while (0)

static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
static float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static void parallel_for(int n, const std::function<void(int)>& fn, int nthreads = 48) {
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  std::exception_ptr err = nullptr;
  std::atomic<bool> failed{false};
  for (int t = 0; t < std::min(n, nthreads); t++)
    th.emplace_back([&] {
      while (true) {
        int i = next++;
        if (i >= n) break;
        try {
          fn(i);
        } catch (...) {
          if (!failed.exchange(true)) err = std::current_exception();
        }
      }
    });
  for (auto& t : th) t.join();
  if (err) std::rethrow_exception(err);
}

// ------------------------------------------------------------------ safetensors
struct STensor {
  const uint16_t* data = nullptr;
  std::vector<int64_t> shape;
};
struct SafeTensors {
  std::map<std::string, STensor> t;
  std::vector<std::pair<void*, size_t>> maps;
  void open_dir(const std::string& dir) {
    std::string idx = dir + "/model.safetensors.index.json";
    std::vector<std::string> files;
    FILE* f = fopen(idx.c_str(), "rb");
    if (f) {
      std::string s;
      char buf[65536];
      size_t n;
      while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
      fclose(f);
      JVal j = json_parse(s);
      const JVal* wm = j.get("weight_map");
      if (!wm) throw std::runtime_error("index without weight_map");
      for (auto& kv : wm->obj)
        if (std::find(files.begin(), files.end(), kv.second.s) == files.end()) files.push_back(kv.second.s);
    } else {
      files.push_back("model.safetensors");
    }
    for (auto& fn : files) load_file(dir + "/" + fn);
  }
  void load_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st;
    fstat(fd, &st);
    void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("mmap failed " + path);
    madvise(m, st.st_size, MADV_WILLNEED);
    maps.push_back({m, (size_t)st.st_size});
    uint64_t hl;
    memcpy(&hl, m, 8);
    std::string hs((const char*)m + 8, hl);
    JVal j = json_parse(hs);
    const char* base = (const char*)m + 8 + hl;
    for (auto& kv : j.obj) {
      if (kv.first == "__metadata__") continue;
      const JVal* dt = kv.second.get("dtype");
      if (!dt || dt->s != "BF16") throw std::runtime_error("unexpected dtype for " + kv.first);
      STensor s;
      for (auto& d : kv.second.get("shape")->arr) s.shape.push_back(d.inum);
      const JVal* off = kv.second.get("data_offsets");
      s.data = (const uint16_t*)(base + off->arr[0].inum);
      t[kv.first] = s;
    }
  }
  const STensor& get(const std::string& n) const {
    auto it = t.find(n);
    if (it == t.end()) throw std::runtime_error("missing tensor " + n);
    return it->second;
  }
};

// lay: dst[p][(nt*KT + kt)*128 + j] = row(nt*128 + j)[kt*128 + p]  (one core's run, [128, NT*KT*128])
static void lay(uint16_t* dst, int NT, int KT, const std::function<const uint16_t*(int)>& row) {
  const size_t W = (size_t)NT * KT * 128;
  for (int nt = 0; nt < NT; nt++)
    for (int j = 0; j < 128; j++) {
      const uint16_t* r = row(nt * 128 + j);
      for (int kt = 0; kt < KT; kt++) {
        const uint16_t* rk = r + kt * 128;
        uint16_t* d = dst + (size_t)(nt * KT + kt) * 128 + j;
        for (int p = 0; p < 128; p++) d[(size_t)p * W] = rk[p];
      }
    }
}
static void col(uint16_t* dst, const uint16_t* v, int n) {  // [128, n/128]
  int T = n / 128;
  for (int t = 0; t < T; t++)
    for (int p = 0; p < 128; p++) dst[p * T + t] = v[t * 128 + p];
}

// ------------------------------------------------------------------ NEFF models and tensors
struct Neff {
  std::string name;
  nrt_model_t* m = nullptr;
  std::map<std::string, size_t> ins, outs;
};

struct Engine {
  int vnc = 0;
  std::map<std::string, Neff> neffs;
  std::vector<int> buckets;  // decode attention buckets (positions read)

  // device tensors
  nrt_tensor_t *embed = nullptr, *lm = nullptr, *gf = nullptr, *idb = nullptr, *consts = nullptr, *cosT = nullptr,
               *sinT = nullptr;
  nrt_tensor_t *toks = nullptr, *junk = nullptr, *P[2] = {nullptr, nullptr}, *Hb[2] = {nullptr, nullptr};
  struct Layer {
    nrt_tensor_t *g1, *wqkv, *wo, *g2, *wgu, *wd, *kc, *vc;
  };
  std::vector<Layer> L;

  // tensor sets
  nrt_tensor_set_t *emb_in[2], *emb_out;
  nrt_tensor_set_t* lay_in[NL][2];
  nrt_tensor_set_t* lay_out[2];
  nrt_tensor_set_t *head_in[2][2], *head_out[2];  // [junk][par], [par]
  // multi-layer decode NEFFs dec_grp<G>_<B> (G consecutive layers per execution), if built
  int grpG = 0;
  std::map<int, std::string> grp_neff;  // bucket -> NEFF name
  std::vector<nrt_tensor_set_t*> grp_in[2];
  nrt_tensor_set_t* grp_out[2];
  // fused decode step: dec_grp16e_<B> (embed + layers 0-15) and dec_grp16h_<B> (layers 16-31 + head)
  std::map<int, std::string> fuse_e, fuse_h;
  nrt_tensor_set_t *fe_in[2], *fe_out, *fh_in[2][2], *fh_out[2];  // fh_in [junk][par]

  // prefill (per prompt bucket S)
  std::vector<int> pf_buckets;
  nrt_tensor_t *cosP = nullptr, *sinP = nullptr;
  struct PF {
    int S;
    nrt_tensor_t *T[2] = {nullptr, nullptr};
    nrt_tensor_t *qs2 = nullptr, *att2 = nullptr, *kq2 = nullptr, *vq2 = nullptr;   // pf_multi pipe: second set
    nrt_tensor_t *X[2], *qs, *att, *kq, *vq, *xa, *xb, *hs, *ms;
    nrt_tensor_set_t *emb_in, *emb_out, *lay_in[NL], *lay_out[2], *pick_in, *pick_out;
    nrt_tensor_set_t *last_in = nullptr, *last_out = nullptr;  // pf_last_S (last layer, last token only)
    nrt_tensor_set_t *first_in = nullptr, *first_out = nullptr;  // pf_first_S (pf_embed + layer 0)
    std::string multi;                                           // pf_multi<G>_S: G layers per execution
    int multiG = 0;
    std::vector<nrt_tensor_set_t*> multi_in, multi_out;          // one per group of layers 1 .. NL-2
    nrt_tensor_set_t *eh_in[2] = {nullptr, nullptr}, *eh_out[2] = {nullptr, nullptr};  // pf_multi16e / pf_multi15h
    nrt_tensor_set_t *lasth_in = nullptr, *lasth_out = nullptr;  // pf_lasth_S (pf_last_S + dec_head in one NEFF)
  };
  std::vector<PF> pf;

  int cur = -1;  // position of the next token to consume (state); -1: no state
  bool loaded = false;

  nrt_tensor_t* alloc(const char* name, size_t bytes) {
    nrt_tensor_t* t;
    CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, vnc, bytes, name, &t));
    CK(nrt_tensor_memset(t, 0, 0, bytes));
    return t;
  }
  void write(nrt_tensor_t* t, const void* p, size_t n, size_t off = 0) { CK(nrt_tensor_write(t, p, off, n)); }

  void load_neffs() {
    for (int i = 0; i < g_nneffs; i++) {
      Neff N;
      N.name = g_neffs[i].name;
      size_t sz = g_neffs[i].end - g_neffs[i].start;
      auto t0 = clk::now();
      CK(nrt_load(g_neffs[i].start, sz, vnc, 1, &N.m));
      nrt_tensor_info_array_t* info;
      CK(nrt_get_model_tensor_info(N.m, &info));
      for (uint64_t k = 0; k < info->tensor_count; k++) {
        auto& ti = info->tensor_array[k];
        (ti.usage == NRT_TENSOR_USAGE_INPUT ? N.ins : N.outs)[ti.name] = ti.size;
      }
      nrt_free_model_tensor_info(info);
      if (g_verbose) logf("loaded %s (%zu bytes) in %.0f ms\n", N.name.c_str(), sz, ms_since(t0));
      if (N.name.rfind("dec_layer_", 0) == 0) buckets.push_back(atoi(N.name.c_str() + 10));
      if (N.name.rfind("pf_layer_", 0) == 0) pf_buckets.push_back(atoi(N.name.c_str() + 9));
      if (N.name.rfind("dec_grp", 0) == 0) {
        int G = atoi(N.name.c_str() + 7);
        size_t u = N.name.find('_', 7);
        char kind = N.name[u - 1];  // 'e': embed + first layer group, 'h': last layer group + head
        if (kind == 'e' || kind == 'h') {
          (kind == 'e' ? fuse_e : fuse_h)[atoi(N.name.c_str() + u + 1)] = N.name;
          neffs[N.name] = N;
          continue;
        }
        if (grpG && G != grpG) throw std::runtime_error("mixed dec_grp sizes");
        grpG = G;
        grp_neff[atoi(N.name.c_str() + u + 1)] = N.name;
      }
      neffs[N.name] = N;
    }
    std::sort(buckets.begin(), buckets.end());
    std::sort(pf_buckets.begin(), pf_buckets.end());
    if (buckets.empty() || buckets.back() < TMAX) throw std::runtime_error("no decode bucket covering TMAX");
  }

  nrt_tensor_set_t* make_set(const Neff& N, bool inputs, const std::vector<std::pair<std::string, nrt_tensor_t*>>& ts) {
    nrt_tensor_set_t* s;
    CK(nrt_allocate_tensor_set(&s));
    const auto& want = inputs ? N.ins : N.outs;
    for (auto& kv : ts) {
      auto it = want.find(kv.first);
      if (it == want.end()) throw std::runtime_error(N.name + ": no tensor named " + kv.first);
      size_t have = nrt_tensor_get_size(kv.second);
      if (have != it->second)
        throw std::runtime_error(N.name + ": size mismatch for " + kv.first + " " + std::to_string(have) + " vs " +
                                 std::to_string(it->second));
      CK(nrt_add_tensor_to_tensor_set(s, kv.first.c_str(), kv.second));
    }
    if (ts.size() != want.size()) {
      std::string miss;
      for (auto& w : want) {
        bool f = false;
        for (auto& kv : ts) f |= kv.first == w.first;
        if (!f) miss += " " + w.first;
      }
      throw std::runtime_error(N.name + ": unbound tensors:" + miss + " (bound " + std::to_string(ts.size()) +
                               " of " + std::to_string(want.size()) + ")");
    }
    return s;
  }
  const Neff& neff(const std::string& n) {
    auto it = neffs.find(n);
    if (it == neffs.end()) throw std::runtime_error("missing NEFF " + n);
    return it->second;
  }

  void load(const std::string& dir) {
    auto t0 = clk::now();
    CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
    load_neffs();
    logf("NEFFs loaded: %.0f ms\n", ms_since(t0));
    SafeTensors st;
    st.open_dir(dir);

    // ---- per-layer weights: re-layout in parallel, then upload
    L.resize(NL);
    const size_t QKV_E = (size_t)2 * 128 * QKV_NT * HT * 128, WO_E = (size_t)2 * 128 * HT * 16 * 128,
                 WGU_E = (size_t)2 * 128 * IT * HT * 128, WD_E = (size_t)2 * 128 * HT * ITC * 128;
    const size_t KC_B = (size_t)8 * 128 * TMAX * 2, VC_B = (size_t)TMAX * 1024 * 2;
    for (int l = 0; l < NL; l++) {
      char nm[64];
      auto& Ly = L[l];
      snprintf(nm, sizeof nm, "g1_%d", l); Ly.g1 = alloc(nm, 128 * HT * 2);
      snprintf(nm, sizeof nm, "g2_%d", l); Ly.g2 = alloc(nm, 128 * HT * 2);
      snprintf(nm, sizeof nm, "wqkv_%d", l); Ly.wqkv = alloc(nm, QKV_E * 2);
      snprintf(nm, sizeof nm, "wo_%d", l); Ly.wo = alloc(nm, WO_E * 2);
      snprintf(nm, sizeof nm, "wgu_%d", l); Ly.wgu = alloc(nm, WGU_E * 2);
      snprintf(nm, sizeof nm, "wd_%d", l); Ly.wd = alloc(nm, WD_E * 2);
      snprintf(nm, sizeof nm, "kc_%d", l); Ly.kc = alloc(nm, KC_B);
      snprintf(nm, sizeof nm, "vc_%d", l); Ly.vc = alloc(nm, VC_B);
    }
    const int LB = 8;  // layers per batch (host buffers)
    for (int l0 = 0; l0 < NL; l0 += LB) {
      int nb = std::min(LB, NL - l0);
      std::vector<std::vector<uint16_t>> bq(nb), bo(nb), bgu(nb), bd(nb);
      parallel_for(nb * 4, [&](int t) {
        int b = t / 4;
        switch (t % 4) {
          case 0: bq[b].resize(QKV_E); break;
          case 1: bo[b].resize(WO_E); break;
          case 2: bgu[b].resize(WGU_E); break;
          default: bd[b].resize(WD_E);
        }
      });
      parallel_for(nb * 9, [&](int task) {
        int b = task / 9, part = task % 9, l = l0 + b;
        std::string p = "model.layers." + std::to_string(l) + ".";
        if (part < 2) {
          int c = part;
          const uint16_t* q = st.get(p + "self_attn.q_proj.weight").data;
          const uint16_t* k = st.get(p + "self_attn.k_proj.weight").data;
          const uint16_t* v = st.get(p + "self_attn.v_proj.weight").data;
          lay(bq[b].data() + (size_t)c * QKV_E / 2, QKV_NT, HT, [&](int n) -> const uint16_t* {
            if (n < 2048) return q + (size_t)(2048 * c + n) * H;
            if (n < 2560) return k + (size_t)(512 * c + n - 2048) * H;
            return v + (size_t)(512 * c + n - 2560) * H;
          });
        } else if (part < 4) {
          int c = part - 2;
          const uint16_t* o = st.get(p + "self_attn.o_proj.weight").data;
          lay(bo[b].data() + (size_t)c * WO_E / 2, HT, 16, [&](int n) -> const uint16_t* { return o + (size_t)n * H + 2048 * c; });
        } else if (part < 6) {
          int c = part - 4;
          const uint16_t* g = st.get(p + "mlp.gate_proj.weight").data;
          const uint16_t* u = st.get(p + "mlp.up_proj.weight").data;
          lay(bgu[b].data() + (size_t)c * WGU_E / 2, IT, HT, [&](int n) -> const uint16_t* {
            int gi = n / (2 * GU_G * 128), r = n % (2 * GU_G * 128);
            int row = (ITC * c + GU_G * gi) * 128 + (r % (GU_G * 128));
            return (r < GU_G * 128 ? g : u) + (size_t)row * H;
          });
        } else if (part < 8) {
          int c = part - 6;
          const uint16_t* d = st.get(p + "mlp.down_proj.weight").data;
          lay(bd[b].data() + (size_t)c * WD_E / 2, HT, ITC, [&](int n) -> const uint16_t* { return d + (size_t)n * 14336 + 7168 * c; });
        } else {
          std::vector<uint16_t> g(128 * HT);
          col(g.data(), st.get(p + "input_layernorm.weight").data, H);
          write(L[l].g1, g.data(), g.size() * 2);
          col(g.data(), st.get(p + "post_attention_layernorm.weight").data, H);
          write(L[l].g2, g.data(), g.size() * 2);
        }
      });
      for (int b = 0; b < nb; b++) {
        int l = l0 + b;
        write(L[l].wqkv, bq[b].data(), QKV_E * 2);
        write(L[l].wo, bo[b].data(), WO_E * 2);
        write(L[l].wgu, bgu[b].data(), WGU_E * 2);
        write(L[l].wd, bd[b].data(), WD_E * 2);
      }
      if (g_verbose) logf("layers %d..%d uploaded: %.0f ms\n", l0, l0 + nb - 1, ms_since(t0));
    }

    // ---- embedding (original layout), LM head, final norm, constants
    {
      const STensor& e = st.get("model.embed_tokens.weight");
      embed = alloc("embed", (size_t)VOCAB * H * 2);
      write(embed, e.data, (size_t)VOCAB * H * 2);
      const size_t LM_E = (size_t)2 * 128 * VTCMAX * HT * 128;
      std::vector<uint16_t> b(LM_E);
      const uint16_t* w = st.get("lm_head.weight").data;
      parallel_for(2 * 8, [&](int task) {
        int c = task / 8, part = task % 8;
        // split the core's 501 n-tiles into 8 contiguous ranges written independently
        const int nc = c == 0 ? VTC - HSK : VTC + HSK, g0 = c == 0 ? 0 : VTC - HSK;
        int n0 = part * nc / 8, n1 = (part + 1) * nc / 8;
        const size_t Wc = (size_t)VTCMAX * HT * 128;
        uint16_t* dst = b.data() + (size_t)c * LM_E / 2;
        for (int nt = n0; nt < n1; nt++)
          for (int j = 0; j < 128; j++) {
            const uint16_t* r = w + (size_t)((g0 + nt) * 128 + j) * H;
            for (int kt = 0; kt < HT; kt++)
              for (int p = 0; p < 128; p++) dst[(size_t)p * Wc + (size_t)(nt * HT + kt) * 128 + j] = r[kt * 128 + p];
          }
      });
      lm = alloc("lm", LM_E * 2);
      write(lm, b.data(), LM_E * 2);
      std::vector<uint16_t> g(128 * HT);
      col(g.data(), st.get("model.norm.weight").data, H);
      gf = alloc("gf", g.size() * 2);
      write(gf, g.data(), g.size() * 2);
    }
    {
      std::vector<float> ib(2 * 128);
      for (int c = 0; c < 2; c++)
        for (int p = 0; p < 128; p++) ib[c * 128 + p] = (float)(c * (VTC - HSK) * 128 + p);
      idb = alloc("idb", ib.size() * 4);
      write(idb, ib.data(), ib.size() * 4);
      std::vector<uint16_t> cs(128 * 256, 0);
      for (int i = 0; i < 128; i++) {
        cs[((i + 64) % 128) * 256 + i] = f2bf(1.f);
        cs[i * 256 + 128 + i] = f2bf(1.f);
      }
      consts = alloc("consts", cs.size() * 2);
      write(consts, cs.data(), cs.size() * 2);
      // RoPE (llama3 scaling), fp32 frequencies as NxD, cos/sin rounded to bf16 (as the reference applies them)
      std::vector<float> inv(64);
      for (int i = 0; i < 64; i++) {
        float e = (float)(2 * i) / 128.f;
        float f = 1.f / powf(500000.f, e);
        float wl = (float)(2 * M_PI) / f;
        if (wl < 2048.f) inv[i] = f;
        else if (wl > 8192.f) inv[i] = f / 8.f;
        else {
          float s = (8192.f / wl - 1.f) / 3.f;
          inv[i] = (1.f - s) * f / 8.f + s * f;
        }
      }
      std::vector<float> c(TMAX * 128), s(TMAX * 128);
      for (int p = 0; p < TMAX; p++)
        for (int d = 0; d < 128; d++) {
          float a = (float)p * inv[d % 64];
          c[p * 128 + d] = bf2f(f2bf((float)cos((double)a)));
          float sv = bf2f(f2bf((float)sin((double)a)));
          s[p * 128 + d] = d < 64 ? -sv : sv;
        }
      std::vector<float> ct(128 * TMAX), stt(128 * TMAX);
      for (int p = 0; p < TMAX; p++)
        for (int d = 0; d < 128; d++) ct[d * TMAX + p] = c[p * 128 + d], stt[d * TMAX + p] = s[p * 128 + d];
      cosP = alloc("cosP", ct.size() * 4);
      write(cosP, ct.data(), ct.size() * 4);
      sinP = alloc("sinP", stt.size() * 4);
      write(sinP, stt.data(), stt.size() * 4);
      cosT = alloc("cosT", c.size() * 4);
      write(cosT, c.data(), c.size() * 4);
      sinT = alloc("sinT", s.size() * 4);
      write(sinT, s.data(), s.size() * 4);
    }
    toks = alloc("toks", TOKS_LEN * 4);
    junk = alloc("junk", TOKS_LEN * 4);
    P[0] = alloc("P0", 4);
    P[1] = alloc("P1", 4);
    Hb[0] = alloc("H0", 128 * HT * 2);
    Hb[1] = alloc("H1", 128 * HT * 2);
    for (auto& m : st.maps) munmap(m.first, m.second);
    logf("weights uploaded: %.0f ms\n", ms_since(t0));

    // ---- tensor sets
    const Neff& E = neff("dec_embed");
    const Neff& LY = neff("dec_layer_" + std::to_string(buckets[0]));
    const Neff& HD = neff("dec_head");
    for (int par = 0; par < 2; par++) {
      emb_in[par] = make_set(E, true, {{"toks", toks}, {"pos", P[par]}, {"embed", embed}});
      for (int l = 0; l < NL; l++)
        lay_in[l][par] = make_set(LY, true, {{"h", Hb[l % 2]}, {"pos", P[par]}, {"cosT", cosT}, {"sinT", sinT},
                                             {"consts", consts}, {"g1", L[l].g1}, {"wqkv", L[l].wqkv}, {"wo", L[l].wo},
                                             {"g2", L[l].g2}, {"wgu", L[l].wgu}, {"wd", L[l].wd}, {"kc", L[l].kc},
                                             {"vc", L[l].vc}});
      lay_out[par] = make_set(LY, false, {{"output_0", Hb[1 - par]}});
      for (int j = 0; j < 2; j++)
        head_in[j][par] = make_set(HD, true, {{"h", Hb[0]}, {"pos", P[par]}, {"toks", j ? junk : toks}, {"gf", gf},
                                              {"lm", lm}, {"idb", idb}});
      head_out[par] = make_set(HD, false, {{"output_0", P[1 - par]}});
    }
    emb_out = make_set(E, false, {{"output_0", Hb[0]}});
    if (grpG) {
      if (NL % (2 * grpG)) throw std::runtime_error("dec_grp size must divide NL/2");
      const Neff& GN = neff(grp_neff.begin()->second);
      for (int par = 0; par < 2; par++)
        for (int gi = 0; gi < NL / grpG; gi++) {
          std::vector<std::pair<std::string, nrt_tensor_t*>> b = {
              {"h", Hb[gi % 2]}, {"pos", P[par]}, {"cosT", cosT}, {"sinT", sinT}, {"consts", consts}};
          for (int j = 0; j < grpG; j++) {
            int l = gi * grpG + j;
            std::string sx = "_" + std::to_string(j);
            b.insert(b.end(), {{"g1" + sx, L[l].g1}, {"wqkv" + sx, L[l].wqkv}, {"wo" + sx, L[l].wo},
                               {"g2" + sx, L[l].g2}, {"wgu" + sx, L[l].wgu}, {"wd" + sx, L[l].wd},
                               {"kc" + sx, L[l].kc}, {"vc" + sx, L[l].vc}});
          }
          grp_in[par].push_back(make_set(GN, true, b));
        }
      for (int j = 0; j < 2; j++) grp_out[j] = make_set(GN, false, {{"output_0", Hb[1 - j]}});
      for (auto& kv : grp_neff)
        if (neff(kv.second).ins != GN.ins || neff(kv.second).outs != GN.outs)
          throw std::runtime_error("dec_grp NEFFs differ in I/O");
    }
    if (!fuse_e.empty()) {
      if (NL != 32 || fuse_e.size() != fuse_h.size()) throw std::runtime_error("fused decode NEFFs incomplete");
      const Neff& FE = neff(fuse_e.begin()->second);
      const Neff& FH = neff(fuse_h.begin()->second);
      auto lays = [&](int l0, std::vector<std::pair<std::string, nrt_tensor_t*>>& b) {
        for (int j = 0; j < 16; j++) {
          int l = l0 + j;
          std::string sx = "_" + std::to_string(j);
          b.insert(b.end(), {{"g1" + sx, L[l].g1}, {"wqkv" + sx, L[l].wqkv}, {"wo" + sx, L[l].wo},
                             {"g2" + sx, L[l].g2}, {"wgu" + sx, L[l].wgu}, {"wd" + sx, L[l].wd},
                             {"kc" + sx, L[l].kc}, {"vc" + sx, L[l].vc}});
        }
      };
      for (int par = 0; par < 2; par++) {
        std::vector<std::pair<std::string, nrt_tensor_t*>> b = {
            {"toks", toks}, {"pos", P[par]}, {"embed", embed}, {"cosT", cosT}, {"sinT", sinT}, {"consts", consts}};
        lays(0, b);
        fe_in[par] = make_set(FE, true, b);
        for (int j = 0; j < 2; j++) {
          std::vector<std::pair<std::string, nrt_tensor_t*>> c = {
              {"h", Hb[1]}, {"pos", P[par]}, {"cosT", cosT}, {"sinT", sinT}, {"consts", consts}};
          lays(16, c);
          c.insert(c.end(), {{"toks", j ? junk : toks}, {"gf", gf}, {"lm", lm}, {"idb", idb}});
          fh_in[j][par] = make_set(FH, true, c);
        }
        fh_out[par] = make_set(FH, false, {{"output_0", P[1 - par]}});
      }
      fe_out = make_set(FE, false, {{"output_0", Hb[1]}});
      for (auto& kv : fuse_e)
        if (neff(kv.second).ins != FE.ins || neff(kv.second).outs != FE.outs ||
            neff(fuse_h.at(kv.first)).ins != FH.ins || neff(fuse_h.at(kv.first)).outs != FH.outs)
          throw std::runtime_error("fused NEFFs differ in I/O");
    }
    for (int S : pf_buckets) {
      PF F;
      F.S = S;
      std::string ss = std::to_string(S);
      size_t xb = (size_t)128 * HT * S * 2;
      F.X[0] = alloc(("pfX0_" + ss).c_str(), xb);
      F.X[1] = alloc(("pfX1_" + ss).c_str(), xb);
      F.qs = alloc(("pfq_" + ss).c_str(), xb);
      F.att = alloc(("pfa_" + ss).c_str(), xb);
      F.kq = alloc(("pfk_" + ss).c_str(), (size_t)8 * 128 * S * 2);  // attention inputs [8,128,S], [8,S,128]
      F.vq = alloc(("pfv_" + ss).c_str(), (size_t)8 * 128 * S * 2);
      F.xa = alloc(("pfxa_" + ss).c_str(), xb * 4);
      F.xb = alloc(("pfxb_" + ss).c_str(), xb * 4);
      F.hs = alloc(("pfh_" + ss).c_str(), xb);
      F.ms = alloc(("pfm_" + ss).c_str(), (size_t)128 * IT * S * 2);
      const Neff& PE = neff("pf_embed_" + ss);
      const Neff& PL = neff("pf_layer_" + ss);
      const Neff& PP = neff("pf_pick_" + ss);
      F.emb_in = make_set(PE, true, {{"toks", toks}, {"embed", embed}, {"consts", consts}});
      F.emb_out = make_set(PE, false, {{"output_0", F.X[0]}});
      for (int l = 0; l < NL; l++) {
        std::vector<std::pair<std::string, nrt_tensor_t*>> b = {
            {"x", F.X[l % 2]}, {"cosP", cosP}, {"sinP", sinP}, {"consts", consts}, {"g1", L[l].g1},
            {"wqkv", L[l].wqkv}, {"wo", L[l].wo}, {"g2", L[l].g2}, {"wgu", L[l].wgu}, {"wd", L[l].wd},
            {"kc", L[l].kc}, {"vc", L[l].vc}, {"qs", F.qs}, {"att", F.att}, {"kq", F.kq}, {"vq", F.vq}};
        // exchange scratch: xa/xb fp32 partials (head/intermediate split; xa also holds the N-split variant's
        // partial sums of squares), hs/ms bf16 activation halves (N-split variant)
        for (auto& kv : std::vector<std::pair<std::string, nrt_tensor_t*>>{
                 {"xa", F.xa}, {"xb", F.xb}, {"hs", F.hs}, {"ms", F.ms}})
          if (PL.ins.count(kv.first)) b.push_back(kv);
        F.lay_in[l] = make_set(PL, true, b);
      }
      for (int j = 0; j < 2; j++) F.lay_out[j] = make_set(PL, false, {{"output_0", F.X[1 - j]}});
      F.pick_in = make_set(PP, true, {{"x", F.X[0]}, {"pos", P[0]}});
      F.pick_out = make_set(PP, false, {{"output_0", Hb[0]}});
      if (neffs.count("pf_last_" + ss) && !getenv("ENGINE_NO_PF_LAST")) {
        const Neff& PLL = neff("pf_last_" + ss);
        int l = NL - 1;
        F.last_in = make_set(PLL, true, {{"x", F.X[l % 2]}, {"cosP", cosP}, {"sinP", sinP}, {"consts", consts},
                                         {"g1", L[l].g1}, {"wqkv", L[l].wqkv}, {"wo", L[l].wo}, {"g2", L[l].g2},
                                         {"wgu", L[l].wgu}, {"wd", L[l].wd}, {"kc", L[l].kc}, {"vc", L[l].vc},
                                         {"qs", F.qs}, {"att", F.att}, {"kq", F.kq}, {"vq", F.vq}, {"hs", F.hs},
                                         {"ms", F.ms}, {"pos", P[0]}});
        F.last_out = make_set(PLL, false, {{"output_0", Hb[0]}});
      }
      if (neffs.count("pf_first_" + ss) && !getenv("ENGINE_NO_PF_FIRST")) {   // embed + layer 0 in one NEFF
        const Neff& PF0 = neff("pf_first_" + ss);
        std::vector<std::pair<std::string, nrt_tensor_t*>> b = {
            {"toks", toks}, {"embed", embed}, {"cosP", cosP}, {"sinP", sinP}, {"consts", consts}, {"g1", L[0].g1},
            {"wqkv", L[0].wqkv}, {"wo", L[0].wo}, {"g2", L[0].g2}, {"wgu", L[0].wgu}, {"wd", L[0].wd},
            {"kc", L[0].kc}, {"vc", L[0].vc}, {"qs", F.qs}, {"att", F.att}, {"kq", F.kq}, {"vq", F.vq}};
        for (auto& kv : std::vector<std::pair<std::string, nrt_tensor_t*>>{
                 {"xa", F.xa}, {"xb", F.xb}, {"hs", F.hs}, {"ms", F.ms}})
          if (PF0.ins.count(kv.first)) b.push_back(kv);
        F.first_in = make_set(PF0, true, b);
        F.first_out = make_set(PF0, false, {{"output_0", F.X[1]}});
      }
      for (auto& kv : neffs) {   // pf_multi<G>_<S>: layers 1 .. NL-2 in groups of G (needs pf_first and pf_lasth)
        const std::string& nm = kv.first;
        if (nm.rfind("pf_multi", 0) != 0 || getenv("ENGINE_NO_PF_MULTI")) continue;
        size_t u = nm.find('_', 8);
        if (u == std::string::npos || nm.substr(u + 1) != ss) continue;
        if (nm.find_first_not_of("0123456789", 8) != u) continue;   // pf_multi<G>_<S> only (not 16e / 15h)
        int G = atoi(nm.c_str() + 8);
        if (G <= 0 || (NL - 2) % G || !neffs.count("pf_lasth_" + ss)) continue;
        F.multi = nm;
        F.multiG = G;
        F.T[0] = alloc(("pfT0_" + ss).c_str(), xb);
        F.T[1] = alloc(("pfT1_" + ss).c_str(), xb);
        const Neff& PM = neff(nm);
        for (int g = 0; g < (NL - 2) / G; g++) {
          int l0 = 1 + g * G;
          std::vector<std::pair<std::string, nrt_tensor_t*>> b = {
              {"x", F.X[l0 % 2]}, {"t0", F.T[0]}, {"t1", F.T[1]}, {"cosP", cosP}, {"sinP", sinP}, {"consts", consts}};
          if (PM.ins.count("qs0")) {   // pipe variant: two scratch sets, alternating per layer
            if (!F.qs2) {
              F.qs2 = alloc(("pfq2_" + ss).c_str(), xb);
              F.att2 = alloc(("pfa2_" + ss).c_str(), xb);
              F.kq2 = alloc(("pfk2_" + ss).c_str(), (size_t)8 * 128 * S * 2);
              F.vq2 = alloc(("pfv2_" + ss).c_str(), (size_t)8 * 128 * S * 2);
            }
            b.insert(b.end(), {{"qs0", F.qs}, {"att0", F.att}, {"kq0", F.kq}, {"vq0", F.vq}, {"qs1", F.qs2},
                               {"att1", F.att2}, {"kq1", F.kq2}, {"vq1", F.vq2}});
          } else {
            b.insert(b.end(), {{"qs", F.qs}, {"att", F.att}, {"kq", F.kq}, {"vq", F.vq}, {"hs", F.hs}, {"ms", F.ms},
                               {"xa", F.xa}});
          }
          for (int j = 0; j < G; j++) {
            int l = l0 + j;
            std::string sx = "_" + std::to_string(j);
            b.insert(b.end(), {{"g1" + sx, L[l].g1}, {"wqkv" + sx, L[l].wqkv}, {"wo" + sx, L[l].wo},
                               {"g2" + sx, L[l].g2}, {"wgu" + sx, L[l].wgu}, {"wd" + sx, L[l].wd},
                               {"kc" + sx, L[l].kc}, {"vc" + sx, L[l].vc}});
          }
          F.multi_in.push_back(make_set(PM, true, b));
          F.multi_out.push_back(make_set(PM, false, {{"output_0", F.X[(l0 + G) % 2]}}));
        }
      }
      if (neffs.count("pf_multi16e_" + ss) && neffs.count("pf_multi15h_" + ss) && NL == 32 &&
          !getenv("ENGINE_NO_PF_EH")) {   // the whole prefill in 2 executions
        const Neff& ME = neff("pf_multi16e_" + ss);
        const Neff& MH = neff("pf_multi15h_" + ss);
        if (!F.T[0]) {
          F.T[0] = alloc(("pfT0_" + ss).c_str(), xb);
          F.T[1] = alloc(("pfT1_" + ss).c_str(), xb);
        }
        bool pipe = ME.ins.count("qs0") > 0;
        if (pipe && !F.qs2) {
          F.qs2 = alloc(("pfq2_" + ss).c_str(), xb);
          F.att2 = alloc(("pfa2_" + ss).c_str(), xb);
          F.kq2 = alloc(("pfk2_" + ss).c_str(), (size_t)8 * 128 * S * 2);
          F.vq2 = alloc(("pfv2_" + ss).c_str(), (size_t)8 * 128 * S * 2);
        }
        for (int h = 0; h < 2; h++) {
          std::vector<std::pair<std::string, nrt_tensor_t*>> b = {
              {"t0", F.T[0]}, {"t1", F.T[1]}, {"cosP", cosP}, {"sinP", sinP}, {"consts", consts}};
          if (h == 0) b.insert(b.end(), {{"toks", toks}, {"embed", embed}});
          else b.push_back({"x", F.X[0]});
          if (pipe) {
            b.insert(b.end(), {{"qs0", F.qs}, {"att0", F.att}, {"kq0", F.kq}, {"vq0", F.vq}, {"qs1", F.qs2},
                               {"att1", F.att2}, {"kq1", F.kq2}, {"vq1", F.vq2}});
            if (h == 1) b.insert(b.end(), {{"hs", F.hs}, {"ms", F.ms}});
          } else {
            b.insert(b.end(), {{"qs", F.qs}, {"att", F.att}, {"kq", F.kq}, {"vq", F.vq}, {"hs", F.hs}, {"ms", F.ms},
                               {"xa", F.xa}});
          }
          for (int j = 0; j < 16; j++) {
            int l = 16 * h + j;
            std::string sx = "_" + std::to_string(j);
            b.insert(b.end(), {{"g1" + sx, L[l].g1}, {"wqkv" + sx, L[l].wqkv}, {"wo" + sx, L[l].wo},
                               {"g2" + sx, L[l].g2}, {"wgu" + sx, L[l].wgu}, {"wd" + sx, L[l].wd},
                               {"kc" + sx, L[l].kc}, {"vc" + sx, L[l].vc}});
          }
          if (h == 1)
            b.insert(b.end(), {{"pos", P[0]}, {"toks", toks}, {"gf", gf}, {"lm", lm}, {"idb", idb}});
          F.eh_in[h] = make_set(h ? MH : ME, true, b);
          F.eh_out[h] = make_set(h ? MH : ME, false, {{"output_0", h ? P[1] : F.X[0]}});
        }
      }
      if (neffs.count("pf_lasth_" + ss) && !getenv("ENGINE_NO_PF_LASTH")) {
        const Neff& PLH = neff("pf_lasth_" + ss);
        int l = NL - 1;
        F.lasth_in = make_set(PLH, true, {{"x", F.X[l % 2]}, {"cosP", cosP}, {"sinP", sinP}, {"consts", consts},
                                          {"g1", L[l].g1}, {"wqkv", L[l].wqkv}, {"wo", L[l].wo}, {"g2", L[l].g2},
                                          {"wgu", L[l].wgu}, {"wd", L[l].wd}, {"kc", L[l].kc}, {"vc", L[l].vc},
                                          {"qs", F.qs}, {"att", F.att}, {"kq", F.kq}, {"vq", F.vq}, {"hs", F.hs},
                                          {"ms", F.ms}, {"pos", P[0]}, {"toks", toks}, {"gf", gf}, {"lm", lm},
                                          {"idb", idb}});
        F.lasth_out = make_set(PLH, false, {{"output_0", P[1]}});
      }
      pf.push_back(F);
    }
    for (auto& kv : neffs)
      if (kv.first.rfind("dec_layer_", 0) == 0) {  // all buckets share the tensor names and sizes
        if (kv.second.ins != LY.ins || kv.second.outs != LY.outs) throw std::runtime_error("bucket NEFFs differ in I/O");
      }
    loaded = true;
    logf("engine ready: %.0f ms\n", ms_since(t0));
  }

  // ---------------------------------------------------------------- scheduling
  std::vector<NRT_STATUS> rets;
  std::vector<nrta_seq_t> seqs;
  size_t nq = 0;
  bool steptime = getenv("ENGINE_STEPTIME") != nullptr;  // diagnostics: wait for and log every execution
  void sched(nrt_model_t* m, nrt_tensor_set_t* in, nrt_tensor_set_t* out) {
    if (steptime) {
      const char* nm = "?";
      for (auto& kv : neffs)
        if (kv.second.m == m) nm = kv.first.c_str();
      static const char* pn = getenv("ENGINE_PROF_NEFF");   // diagnostics: profile one execution (real data)
      static int pidx = getenv("ENGINE_PROF_IDX") ? atoi(getenv("ENGINE_PROF_IDX")) : 3;
      static std::map<std::string, int> cnt;
      bool prof = pn && strcmp(pn, nm) == 0 && cnt[nm]++ == pidx;
      if (prof) CK(nrt_profile_start(m, getenv("ENGINE_PROF_OUT")));
      auto t0 = clk::now();
      sched1(m, in, out);
      wait_all();
      if (prof) {
        CK(nrt_profile_stop(getenv("ENGINE_PROF_OUT")));
        logf("profiled %s\n", nm);
      }
      logf("exec %s %.3f ms\n", nm, ms_since(t0));
      return;
    }
    sched1(m, in, out);
  }
  const char* skip = getenv("ENGINE_SKIP");  // diagnostics (timing only, wrong tokens): skip NEFFs by name prefix
  void sched1(nrt_model_t* m, nrt_tensor_set_t* in, nrt_tensor_set_t* out) {
    if (skip)
      for (auto& kv : neffs)
        if (kv.second.m == m) {
          std::string sk = std::string(",") + skip + ",";
          for (size_t a = 1, b; (b = sk.find(',', a)) != std::string::npos; a = b + 1)
            if (b > a && kv.first.compare(0, b - a, sk, a, b - a) == 0) return;
        }
    if (nq >= rets.size()) throw std::runtime_error("schedule overflow");
    while (true) {
      NRT_STATUS r = nrta_execute_schedule(m, in, out, 0, &rets[nq], &seqs[nq]);
      if (r == NRT_SUCCESS) break;
      if (r != NRT_QUEUE_FULL) CK(r);
      // wait for the oldest outstanding one
      for (size_t k = 0; k < nq; k++) {
        bool done = false;
        CK(nrta_is_completed(seqs[k], &done));
        if (!done) break;
      }
    }
    nq++;
  }
  void wait_all() {
    if (!nq) return;
    bool done = false;
    while (!done) CK(nrta_is_completed(seqs[nq - 1], &done));
    for (size_t k = 0; k < nq; k++)
      if (rets[k] != NRT_SUCCESS) {
        // earlier ones may still be in flight only if out of order; poll them too
        bool d2 = false;
        while (!d2) CK(nrta_is_completed(seqs[k], &d2));
        if (rets[k] != NRT_SUCCESS) throw std::runtime_error("execution failed: " + std::to_string((int)rets[k]));
      }
    nq = 0;
  }
  int bucket_for(int npos) {
    for (int b : buckets)
      if (b >= npos) return b;
    throw std::runtime_error("position beyond compiled buckets");
  }
  // decode steps at positions p0 .. p0+n-1; the head writes toks[p+1] unless junk(step) is set
  void decode_steps(int p0, int n, int njunk) {
    if (n <= 0) return;
    size_t need = (size_t)n * (NL + 2);
    if (rets.size() < need) {
      rets.resize(need);
      seqs.resize(need);
    }
    int32_t pv = p0;
    write(P[0], &pv, 4);
    nq = 0;
    for (int s = 0; s < n; s++) {
      int par = s & 1, p = p0 + s;
      int b = bucket_for(p + 1);
      auto fe = fuse_e.find(b);
      if (fe != fuse_e.end() && !getenv("ENGINE_NO_FUSE")) {
        static const char* dp = getenv("ENGINE_DEC_PATTERN");   // diagnostics (timing only): "ee", "hh"
        if (dp && dp[0] == 'e') {
          sched(neff(fe->second).m, fe_in[par], fe_out);
          sched(neff(fe->second).m, fe_in[par], fe_out);
          continue;
        }
        if (dp && dp[0] == 'h') {
          sched(neff(fuse_h.at(b)).m, fh_in[s < njunk ? 1 : 0][par], fh_out[par]);
          sched(neff(fuse_h.at(b)).m, fh_in[s < njunk ? 1 : 0][par], fh_out[par]);
          continue;
        }
        sched(neff(fe->second).m, fe_in[par], fe_out);
        sched(neff(fuse_h.at(b)).m, fh_in[s < njunk ? 1 : 0][par], fh_out[par]);
        continue;
      }
      sched(neff("dec_embed").m, emb_in[par], emb_out);
      auto gq = grp_neff.find(b);
      if (gq != grp_neff.end() && !getenv("ENGINE_NO_GRP")) {
        nrt_model_t* gm = neff(gq->second).m;
        for (int gi = 0; gi < NL / grpG; gi++) sched(gm, grp_in[par][gi], grp_out[gi % 2]);
      } else {
        const Neff& LY = neff("dec_layer_" + std::to_string(b));
        for (int l = 0; l < NL; l++) sched(LY.m, lay_in[l][par], lay_out[l % 2 == 0 ? 0 : 1]);
      }
      sched(neff("dec_head").m, head_in[s < njunk ? 1 : 0][par], head_out[par]);
    }
    wait_all();
  }

  std::vector<int> generate(const std::vector<int>& prompt, int max_tokens, bool reset) {
    if (max_tokens < 0) throw std::runtime_error("max_tokens < 0");
    std::vector<int> out;
    if (reset) {
      int Lp = (int)prompt.size();
      if (Lp == 0) throw std::runtime_error("empty prompt with reset");
      if (Lp + max_tokens - 1 > TMAX) throw std::runtime_error("request exceeds the compiled positions");
      for (int t : prompt)
        if (t < 0 || t >= VOCAB) throw std::runtime_error("token id out of range");
      cur = -1;
      std::vector<int32_t> tv(prompt.begin(), prompt.end());
      auto tw0 = clk::now();
      write(toks, tv.data(), tv.size() * 4);
      if (g_timing) logf("timing: toks write %.3f ms\n", ms_since(tw0));
      if (max_tokens == 0) {  // consume the prompt only: state = all but nothing emitted
        throw std::runtime_error("max_tokens 0 not supported");
      }
      if (Lp == 1) {
        // The reference (NxD context encoding) pads position_ids with 1 and takes the output row at
        // argmax(position_ids): for a one-token prompt that is the first pad row, i.e. its first token is the
        // prediction after [prompt, pad_token_id=0] at position 1. Then it decodes from position 1 as usual.
        int32_t pad = 0;
        write(toks, &pad, 4, 4);
        prefill(2);
        int32_t first;
        CK(nrt_tensor_read(toks, &first, 8, 4));
        write(toks, &first, 4, 4);
      } else {
        prefill(Lp);
      }
      decode_steps(Lp, max_tokens - 1, 0);
      std::vector<int32_t> r(max_tokens);
      CK(nrt_tensor_read(toks, r.data(), (size_t)Lp * 4, (size_t)max_tokens * 4));
      out.assign(r.begin(), r.end());
      cur = Lp + max_tokens - 1;
    } else {
      if (cur < 0) throw std::runtime_error("continue without state");
      if (!prompt.empty()) throw std::runtime_error("continue with a non-empty prompt");
      if (max_tokens == 0) return out;
      if (cur + max_tokens > TMAX) throw std::runtime_error("request exceeds the compiled positions");
      decode_steps(cur, max_tokens, 0);
      std::vector<int32_t> r(max_tokens);
      CK(nrt_tensor_read(toks, r.data(), (size_t)(cur + 1) * 4, (size_t)max_tokens * 4));
      out.assign(r.begin(), r.end());
      cur += max_tokens;
    }
    for (int t : out)
      if (t < 0 || t >= VOCAB) throw std::runtime_error("device produced an invalid token");
    return out;
  }

  // consume prompt toks[0..Lp-1]; leaves toks[Lp] = first generated token
  void prefill(int Lp) {
    if (pf.empty() || getenv("ENGINE_PREFILL_BY_DECODE")) {  // debugging path: every prompt token through decode
      decode_steps(0, Lp, Lp - 1);
      return;
    }
    const PF* F = nullptr;
    for (auto& f : pf)
      if (f.S >= Lp) { F = &f; break; }
    bool whole = F != nullptr;
    if (!F) F = &pf.back();  // longer prompts: prefill the first S tokens, then decode steps for the rest
    size_t need = NL + 3;
    if (rets.size() < need) {
      rets.resize(need);
      seqs.resize(need);
    }
    int32_t pv = Lp - 1;
    auto ts0 = clk::now();
    write(P[0], &pv, 4);
    if (g_timing) logf("timing: P write %.3f ms\n", ms_since(ts0));
    nq = 0;
    if (const char* pat = getenv("ENGINE_PF_PATTERN")) {   // diagnostics: an execution pattern instead of the prefill
      size_t n = strlen(pat);
      if (rets.size() < n + 4) {
        rets.resize(n + 4);
        seqs.resize(n + 4);
      }
      for (size_t i = 0; i < n; i++) {
        char c = pat[i];
        if (c == 'E') sched(neff("pf_embed_" + std::to_string(F->S)).m, F->emb_in, F->emb_out);
        if (c == 'P') sched(neff("pf_pick_" + std::to_string(F->S)).m, F->pick_in, F->pick_out);
        if (c == 'L') sched(neff("pf_layer_" + std::to_string(F->S)).m, F->lay_in[0], F->lay_out[0]);
        if (c == 'H') sched(neff("dec_head").m, head_in[0][0], head_out[0]);
      }
      wait_all();
      return;
    }
    const Neff& PL = neff("pf_layer_" + std::to_string(F->S));
    if (whole && F->eh_in[0]) {   // pf_multi16e (embed + layers 0-15), pf_multi15h (layers 16-30 + last + head)
      sched(neff("pf_multi16e_" + std::to_string(F->S)).m, F->eh_in[0], F->eh_out[0]);
      sched(neff("pf_multi15h_" + std::to_string(F->S)).m, F->eh_in[1], F->eh_out[1]);
      wait_all();
      return;
    }
    int l0 = 0;
    if (F->first_in) {
      auto tq = clk::now();
      sched(neff("pf_first_" + std::to_string(F->S)).m, F->first_in, F->first_out);
      if (g_timing) logf("timing: first sched %.3f ms\n", ms_since(tq));
      l0 = 1;
    } else {
      sched(neff("pf_embed_" + std::to_string(F->S)).m, F->emb_in, F->emb_out);
    }
    if (whole && F->lasth_in && F->multiG) {   // layer 0 (pf_first or pf_layer), (NL-2)/G x pf_multi, pf_lasth
      if (l0 == 0) sched(PL.m, F->lay_in[0], F->lay_out[0]);
      nrt_model_t* mm = neff(F->multi).m;
      for (size_t g = 0; g < F->multi_in.size(); g++) sched(mm, F->multi_in[g], F->multi_out[g]);
      sched(neff("pf_lasth_" + std::to_string(F->S)).m, F->lasth_in, F->lasth_out);
    } else if (whole && F->lasth_in && Lp >= 128) {  // last layer + head fused
      for (int l = l0; l < NL - 1; l++) sched(PL.m, F->lay_in[l], F->lay_out[l % 2]);
      sched(neff("pf_lasth_" + std::to_string(F->S)).m, F->lasth_in, F->lasth_out);
    } else if (whole && F->last_in) {  // last layer computes only the last prompt token's output (= pf_pick's input)
      for (int l = l0; l < NL - 1; l++) sched(PL.m, F->lay_in[l], F->lay_out[l % 2]);
      sched(neff("pf_last_" + std::to_string(F->S)).m, F->last_in, F->last_out);
      sched(neff("dec_head").m, head_in[0][0], head_out[0]);
    } else {
      for (int l = l0; l < NL; l++) sched(PL.m, F->lay_in[l], F->lay_out[l % 2]);
    }
    if (whole && !F->last_in) {
      sched(neff("pf_pick_" + std::to_string(F->S)).m, F->pick_in, F->pick_out);
      sched(neff("dec_head").m, head_in[0][0], head_out[0]);
    }
    auto ts1 = clk::now();
    wait_all();
    if (g_timing) logf("timing: prefill schedule %.3f ms, wait %.3f ms\n", std::chrono::duration<double, std::milli>(ts1 - ts0).count(), ms_since(ts1));
    if (!whole) decode_steps(F->S, Lp - F->S, Lp - F->S - 1);
  }
};

// ------------------------------------------------------------------ protocol
static int g_out_fd = 1;
static void reply(const std::string& s) {
  std::string l = s + "\n";
  size_t off = 0;
  while (off < l.size()) {
    ssize_t w = ::write(g_out_fd, l.data() + off, l.size() - off);
    if (w <= 0) {
      if (errno == EINTR) continue;
      exit(3);
    }
    off += w;
  }
}
static std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\', o += c;
    else if (c == '\n') o += "\\n";
    else if ((unsigned char)c < 0x20) o += ' ';
    else o += c;
  }
  return o + "\"";
}

// read one line from stdin; spin (non-blocking) for a while after the last input, then block
static bool read_line(std::string& line, std::string& buf) {
  static auto last = clk::now();
  while (true) {
    size_t nl = buf.find('\n');
    if (nl != std::string::npos) {
      line = buf.substr(0, nl);
      buf.erase(0, nl + 1);
      last = clk::now();
      return true;
    }
    struct pollfd pf = {0, POLLIN, 0};
    int to = ms_since(last) < 10000 ? 0 : 100;
    int r = poll(&pf, 1, to);
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (r == 0) continue;
    char tmp[1 << 16];
    ssize_t n = ::read(0, tmp, sizeof tmp);
    if (n < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      return false;
    }
    if (n == 0) {
      if (!buf.empty()) {
        line = buf;
        buf.clear();
        return true;
      }
      return false;
    }
    buf.append(tmp, n);
  }
}

int main(int argc, char** argv) {
  std::string model;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
  }
  if (model.empty()) {
    fprintf(stderr, "usage: %s --model MODEL_DIR\n", argv[0]);
    return 2;
  }
  g_verbose = getenv("ENGINE_VERBOSE") != nullptr;
  // protocol replies go to a private copy of stdout; fd 1 -> stderr (libnrt may print)
  g_out_fd = dup(1);
  dup2(2, 1);
  Engine E;
  std::string line, buf;
  while (read_line(line, buf)) {
    if (line.empty()) continue;
    auto tl = clk::now();
    int64_t id = -1;
    try {
      JVal j = json_parse(line);
      const JVal* idv = j.get("id");
      if (idv) id = idv->inum;
      const JVal* op = j.get("op");
      if (!op) throw std::runtime_error("missing op");
      if (op->s == "hello") {
        if (!E.loaded) E.load(model);
        reply("{\"id\": " + std::to_string(id) + ", \"ok\": true, \"protocol\": 1}");
      } else if (op->s == "generate") {
        if (!E.loaded) E.load(model);
        std::vector<int> prompt;
        if (const JVal* p = j.get("prompt"))
          for (auto& v : p->arr) prompt.push_back((int)v.inum);
        const JVal* mt = j.get("max_tokens");
        const JVal* rs = j.get("reset");
        int n = mt ? (int)mt->inum : 0;
        bool reset = rs ? rs->b : true;
        auto t0 = clk::now();
        std::vector<int> toks = E.generate(prompt, n, reset);
        double tg = ms_since(t0), tp = std::chrono::duration<double, std::milli>(t0 - tl).count();
        std::string r = "{\"id\": " + std::to_string(id) + ", \"ok\": true, \"tokens\": [";
        for (size_t i = 0; i < toks.size(); i++) r += (i ? ", " : "") + std::to_string(toks[i]);
        reply(r + "]}");
        if (g_verbose)
          logf("generate L=%zu n=%d reset=%d: parse %.3f generate %.3f total %.3f ms\n", prompt.size(), n, (int)reset,
               tp, tg, ms_since(tl));
      } else {
        throw std::runtime_error("unknown op " + op->s);
      }
    } catch (std::exception& e) {
      logf("error: %s\n", e.what());
      reply("{\"id\": " + std::to_string(id) + ", \"ok\": false, \"error\": " + jstr(e.what()) + "}");
    }
  }
  return 0;
}
