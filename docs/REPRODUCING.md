# Reconstructing and preparing the recipe

This source release preserves the deployed engine changes and source inputs. It is not a completed fresh-machine installer. The public packaging has not been deployed to a new cluster. The steps below distinguish reproducible source reconstruction from the remaining runtime assembly work.

## 1. Source reconstruction — supported offline

Requires Python 3.10+ and the exact upstream archive. Download it from the public URL in `provenance.json` using your preferred downloader, then:

```sh
python3 scripts/prepare_source.py --archive local/tensorfold-base.tar.gz --out work/engine
```

SHA256 must be `c39fe71f07f6dd911bcfcdc51b47d27d83535d4f22c1f71915e6b12584412add`. This is BertholomusAI/TensorFold revision `fbbc97132cd004237d9887b8d3185896efaedc5c`. Existing output is refused. Every overlay file's upstream and exported hash is checked. Source files carry modification notices and retained copyright headers; `provenance.json` also records their deployed source hashes before those comments.

## 2. Weights and physical layout

Obtain the exact Mia EAQ v1 model and DeepSeek 1017 Engram revisions from `provenance.json`, subject to their model licenses. Do not substitute the earlier Laky weight export or a different quantization. No model weights are in this repository.

The original shape is padded to 72 attention heads, 9 output groups and physical vocabulary 129,408, with logical vocabulary 129,280 still masked in target and draft sampling. Whole 128-element EXL3 Hadamard blocks are preserved. Padded attention groups duplicate a source group with zero downstream scales; there is no dequantize/requantize step.

`tools/export/prepare_checkpoint.py` consumes pinned checkpoint-header metadata and the native loader's per-rank key plans. `tools/export/rank_cache.py` writes those plans from verified source files with bounded reads; by default it only prints a dry run. `--write` performs the export. `--pollard` remains a required legacy argument, but the selected Mia-only plan does not read that source: pass an unused local directory and do not introduce an older model. See each script's `--help` and the offline tests.

The native key plans must be produced by the reconstructed engine's `zig/src/families/dsv41/plan_main.zig`, not guessed or copied from a different world size. Source metadata must include `config.json`, `headers.json` (header lengths/hashes, file sizes and tensor maps), and the safetensors index. This public release does not yet automate fetching all metadata, compiling the plan helper, staging Engram per rank, extracting vision tensors or assembling the finished runtime; those remain explicit packaging gaps.

## 3. Toolchain, kernels and binaries

The deployment used Linux aarch64, Zig 0.17.0, a CUDA-capable build environment, Triton and NCCL. Use the pinned Mia/Bertholomus upstream build instructions and check their container/toolchain requirements; the deployment's local Docker image ID is not a downloadable registry artifact and is intentionally not offered as one.

Required components include:

- A complete Bertholomus base kernel kit, plus the TP3 AOT launch matrix and throughput additions. Base kit source: `bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10`, revision `abe07c7d509a0199cc7fec15263be2aa52ac652f`.
- The SM121 throughput sources and explicit Triton cases under `tools/kernels/`. `compile_triton.py` compiles cases without allocating GPU tensors. These additions alone are **not** the complete base kit.
- Embedded fatbins matching the reconstructed source, including the corrected GPU candidate sampler. The accepted deployed sampler SHA256 was `4446b1178e25222d3cdf11ccdea473cf65c223a0d95271b06f38d27e273914bd`; a different compiler may produce different binary hashes and needs its own qualification.
- Vision attention/kernel assets and 1017 vision bias derived through the upstream tools; dependencies keep their own licenses.

The head was built using the engine's existing entrypoint `zig build native test -Dfatbins=<complete-fatbin-directory> -j1 --summary all`. The follower used `config/worker-build-argv.json`: substitute the five explicit placeholders for the Zig executable, fatbin directory, kernel-options file, output and global cache; run from the reconstructed engine root. `config/worker-kernel-options.zig` retains `with_kernels=true` and `has_dsv41_port=false`. Do not run this argument list as a shell string: use an argv-aware launcher. Build inside your normal bounded build environment; no build or download runs automatically in this repository.

Native CPU tests do not replace AOT launch-matrix coverage, deterministic GPU checks or per-rank artifact verification. Do not start serving with an incomplete kit.

## 4. Runtime layout and launch

On each rank, prepare and verify:

| Config entry | Contents |
|---|---|
| `runtime` | `model/` with padded config, tokenizer/template and vision/index inputs; `dsv41_token_map.json`; required `local-libs/` |
| `kit` | Complete audited kernel kit including `aot/`, cubins and vision assets |
| `rank_cache` | This rank's native TP3 cache files, matching the exact loader keys |
| `engram` | The matching rank's two Engram safetensors inputs |
| `serving` | `tensorfold-native` on rank 0; `tf-dsv41-lanes` on followers |
| `spill` | Dedicated writable ordinary prompt/KV cache; never share it with another topology |

Make a private copy of `config/cluster.example.json` in ignored `local/`. Replace every placeholder. The master address `192.0.2.10` is a documentation example, not a working cluster address. Configure the actual control NIC and RDMA devices on each machine; a three-way link test must pass first. This recipe does not change network configuration.

```sh
python3 scripts/render_launch.py --config local/cluster.json --rank 1
python3 scripts/render_launch.py --config local/cluster.json --rank 2
python3 scripts/render_launch.py --config local/cluster.json --rank 0
```

The renderer only prints shell-quoted commands. Review them, drain other GPU users, then run followers and head on their respective machines using your coordinated lifecycle. Use a locally prepared image pinned by digest. It must provide the required NVIDIA userspace/runtime libraries and device access. Do not treat a running container or open TCP port as successful inference.

The head binds to `127.0.0.1:8000`. Supply and test your own authenticated gateway. The deployed gateway/key policies are private and are not part of this source release. Do not expose the raw listener publicly.

A 2,200 MHz GPU ceiling was used for the reported results. Applying it is an administrator/hardware setting; the renderer does not apply it or change boot services. Keep host-specific power/clock policies explicit in benchmark comparisons.

## 5. Qualification

Run offline export and packaging tests first:

```sh
python3 -B -m unittest discover -s tools/export -p 'test_*.py' -v
python3 -B -m unittest discover -s tests -p 'test_*.py' -v
```

Then, in a controlled ordinary-work window, test all ranks and your gateway: authentication, streaming, required/named/automatic tools and continuation, thinking on/off and low/high/max, supported strict schemas and explicit 422 for unsupported unions, image input, cancellation, output limits and warm/cold cache behavior. Use synthetic input only. Validate concurrent requests against your actual history lengths. Observe after long prefills, not just on a fresh empty server.

Secure/no-retention use is unsupported. Recent TP2 spill files were not migrated in the recorded deployment, and no cross-topology conversion is provided. Keep rollback assets and cache namespaces separate.
