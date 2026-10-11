# Strix Point (Ryzen AI 300, Radeon 890M / 880M, gfx1150): experimental

Strata's setup and HIP build accept the Ryzen AI 300 "Strix Point" APUs (Radeon 890M / 880M, RDNA3.5, `gfx1150`) as an
**experimental** target. **Nobody has run it on a real chip yet**: the maintainers have no Strix Point machine, and this
page records what was changed and what is still a guess. Please report what you see (see the last section).

Strix Point is the smaller sibling of Strix Halo ([STRIX_HALO.md](STRIX_HALO.md)): the same RDNA3.5 wave32 / 64 KiB LDS /
signed-dot4 family, one pool of system memory shared by the CPU and the GPU, a 16-CU iGPU instead of up to 40.

## What setup and the build do for it

- **Detection:** setup recognizes the chip by its PCI id (`1002:150E`) or `gfx1150` exactly (never a prefix: gfx1152 and
  gfx1103 stay unsupported) and treats it as **unified memory**, like Strix Halo: the BIOS carve-out plus the shared pool
  (the GTT, about half of the RAM by default) is the GPU's room, and the expert cache is sized from what the OS can give
  back (`device_free_bytes()`), not from the carve-out.
- **Build:** `gfx1150` is in the HIP build's "unvalidated" list: it builds with a warning. The matrix-core kernels (prompt
  GEMM / attention / scorer / fused experts) already list gfx1150 (`include/strata/kernels/gfx_arch.hpp`).
- **What it does NOT get:** the gfx1151 defaults (`STRATA_GFX1151_DEFAULTS`, [STRIX_HALO.md](STRIX_HALO.md) section 4) are
  measured on Strix Halo only, and there is no hipBLASLt solution table for gfx1150 (the engine accepts a table only for
  its own architecture): the prompt's dense products use plain hipBLAS and setup says so. The model recommendation
  (UD-IQ4_XS from about 80 GB) is Strix Halo's too and is not given here.

## Build by hand (Linux)

The same steps as [STRIX_HALO.md](STRIX_HALO.md), with `gfx1150` for the architecture and without the hipBLASLt table:

```sh
cmake -S . -B build-point -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1150          # plus the compiler / ROCm paths of STRIX_HALO.md section 2
cmake --build build-point --target strata -j 8
```

You need a ROCm whose compiler and libraries support `gfx1150`. `./setup.sh --backend hip` installs one from AMD's TheRock
wheels into `.venv`; the index it uses for gfx1150 (`https://rocm.nightlies.amd.com/v2/gfx1150/`) follows gfx1151's
pattern and **was not checked against the live index**: if the install fails, point `STRATA_ROCM_INDEX` at a working one,
or install ROCm yourself (a system ROCm 7 in `/opt/rocm` with hipcc and hipBLAS is used first when present).

## What to expect, and the open questions

- **Speed:** the 890M shares the system memory's bandwidth with the CPU and has 16 CUs. Expect far lower numbers than the
  discrete cards in the README or Strix Halo's. No measurement exists yet.
- **Registers:** the 890M is reported to have a smaller vector register file per SIMD than gfx1151. Kernels tuned for
  gfx1151's occupancy may spill or run fewer blocks per CU on it. This costs speed, not correctness, but it is unmeasured.
- **Memory:** RAM decides the model (see "Which model should I pick?" in the README). Give the GPU room with a small BIOS
  carve-out and, if needed, the kernel's `ttm.pages_limit` / `ttm.page_pool_size` as in STRIX_HALO.md; setup changes no
  host setting.
- **Windows:** the ready-made AMD engine carries no gfx1150 code, so setup refuses it there; build the engine yourself
  (`tools\hip\build_windows.bat` with `STRATA_HIP_ARCHS=gfx1150`) and pass it with `--prebuilt`.

## Reports

A report helps: the machine (CPU, RAM, BIOS carve-out), the kernel and ROCm versions, the model, the engine log
(`strata-<model>.log`) and the output of `ctest --test-dir build-point` ([COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)).
