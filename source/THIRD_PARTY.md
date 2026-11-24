# Third-party code

## AWS NKI Library (nkilib), Apache-2.0

`kernels/nkattn/` is a copy of part of the NKI Library (github.com/aws-neuron/nki-library, Apache License 2.0,
Copyright Amazon.com, Inc. or its affiliates), as shipped in the neuronx-cc 2.27.5334 wheel
(`site-packages/nkilib/`). License headers kept; the files are unchanged except `attention_cte.py`, which carries
EngineSmith changes since round 003 (each marked `EngineSmith` in the source): one-pass `range_select` masking of
causal diagonal tiles, no score/exp work for keys past a query group's last row (masked anyway), and hardware-DGE
Q/K/V tile loads. Bitwise identical outputs to the stock kernel (tested on device at S = 256, 1024, 4096).

| engine file | nkilib file |
|---|---|
| `kernels/nkattn/attention/attention_cte.py` | `nkilib/core/attention/attention_cte.py` |
| `kernels/nkattn/utils/{allocator,tree_logger,interleave_copy,kernel_assert,kernel_helpers,common_types,logging,modular_allocator,stream_shuffle_broadcast}.py` | `nkilib/core/utils/<same names>.py` |

The `__init__.py` files of `kernels/nkattn/` hold only the license header. Since round 004 the prefill kernels use `kernels/pattn.py` (written for this engine; it re-issues the
instruction sequence of nkilib's `attention_cte` for our configuration, derived from the Apache-2.0 source above,
so the same notice applies); `_attention_cte` (`PF_ATTN = 0`) (causal prefill attention, the kernel the reference NxD Inference uses for context encoding).
It uses only the public NKI API (`nki.language`, `nki.isa`).

## Everything else

The other NKI kernels (`kernels/*.py`) and the host (`host/*.cpp`, `host/json.h`) were written for this engine.
Design techniques (column layout for decode GEMVs, weight-tile re-layout at load time, `.incbin` embedding of NEFFs,
cache writes at a device-held position, multi-layer decode NEFFs) follow the EngineSmith skill's Trainium backend notes.

Build-time tools (not part of the executable): AWS Neuron SDK `nki` 0.6.0 and `neuronx-cc` 2.27.5334 from the public
Neuron repositories, used through NKI's documented NEFF route. Run time links only `libnrt` (Neuron runtime) plus
the C/C++ runtime libraries.
