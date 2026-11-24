#!/usr/bin/env bash
# Build: NKI kernels -> NEFFs (Python/NKI/neuronx-cc at build time only), embedded into one native executable
# target/engine that links only libnrt (+ libc/libstdc++/libm/libgcc_s).
set -euo pipefail
cd "$(dirname "$0")"
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
NRT=/opt/aws/neuron
export PATH=$V/bin:$PATH                 # nki shells out to neuronx-cc
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
# NKI's documented route compiles each kernel by running it once: point it at the run's experiment core.
export NEURON_RT_VISIBLE_CORES=${NEFF_BUILD_CORES:-$(( ${ENGINESMITH_NEURON_CORE:-0} + 2 ))}
mkdir -p target
"$V/bin/python" kernels/build_neffs.py target
g++ -O2 -std=c++17 -I"$NRT/include" -Ihost host/main.cpp target/gen/neff_table.cpp target/gen/neffs.S \
    -o target/engine -L"$NRT/lib" -lnrt -Wl,-rpath,"$NRT/lib"
echo "built target/engine"
