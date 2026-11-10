# DeepSeek V4.1 Flash 1017 on three GB10 Sparks

An **experimental source recipe** combining MiaAI-Lab's **1017 EAQ v1 EXL3 2.9-bpw weights** with TensorFold/BertholomusAI and LakyFx three-rank execution and throughput work. It runs on three 128 GB NVIDIA GB10 systems.

**Credits first:** this is an integration of substantial upstream work, not a new model or quantization. Start with [CREDITS.md](CREDITS.md), [NOTICE](NOTICE) and the original notices under [licenses/](licenses/).

## Status

The integrated engine has run on a three-GB10 deployment. The November 9 source refresh passed 697 CPU tests and bounded live streaming, tools, vision, cache, supported-schema and authentication checks. The configuration uses **12 slots**, a **6.4M-token shared pool**, a **1,048,576-token advertised context window** and **65,536 maximum output tokens**, with a 2,200 MHz GPU ceiling.

**Important limitations:** intermittent slowdowns remain unresolved. A small request after ordinary traffic resumed waited 92 seconds for its first token. Twelve slots do not guarantee twelve simultaneous long-context generations. Near-1M, mixed long-history performance and durable slowdown prevention are not qualified. Reasoning leakage is not proven eliminated. Object-union JSON schemas are refused with HTTP 422 rather than enforced. See [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

This initial source release is **not a one-command installer**. It publishes the exact engine delta, export tools, kernel source inputs, reproducible source reconstruction, configuration and launch-command renderer. Building and assembling a complete runtime still requires the upstream toolchain, model export and kernel kit. A fresh-machine, end-to-end installation of this public packaging has not been tested. No container, weights, cached conversations or prebuilt binaries are bundled.

## Short measured results

| Workload | Decode speed |
|---|---:|
| C1 code | 100.97 tok/s |
| C1 prose | 65.89 tok/s |
| C1 structured text | 147.33 tok/s |
| C4 aggregate | 153.63 tok/s |
| C8 aggregate | 200.92 tok/s |

These are short smoke checks: one repetition per C1 prompt and 20-second concurrent windows, temperature 0, thinking off, prompt set B, output cap 384. Aggregate throughput is **not** each request's speed. No claim of a matched speedup over TP2 or any upstream recipe follows from this table. [Protocol and results](benchmarks/README.md).

Earlier November 5 TP3 configurations produced 156/164 HumanEval and 147/164 HumanEvalPlus-Mini in the retained qualification. Those scores are historical, not a fresh quality run of the November 9 update. No private application evaluation data is included.

## What this combines

- Mia's original 1017 EAQ v1 values, scales and indices, repartitioned by whole EXL3 blocks without requantization or expert pruning.
- Laky/TensorFold three-rank geometry, logical-vocabulary masking, smart graphs, adaptive speculation, row prefill and mid-batch EXL3 work.
- GPU candidate sampling and a measured draft/round cost table from the tested integration.
- Relevant TP2 restore-room, cache-resume, scheduler, CUDA-context, API and incremental tool-streaming corrections, through adopted upstream patches 0064/0065.
- Existing local schema refusal and explicitly reopened-thinking handling. These do not solve every malformed reasoning-output pattern.

KV sharding is **off**. A 6.4M shared pool is not a 6.4M per-request context. Disk spill is ordinary prompt/KV caching and can contain conversation-derived information; this recipe is **not qualified for secure/no-retention chats**.

## Use the source

1. Read [REPRODUCING.md](docs/REPRODUCING.md) and the requirements/known gaps.
2. Fetch the pinned upstream archive listed in `provenance.json` and reconstruct the engine with `scripts/prepare_source.py`. The script verifies the archive and every changed source file; it does not build or run anything.
3. Use the retained export tools and upstream build tooling to prepare the per-rank weights, complete AOT/fatbin kit, model metadata, vision and Engram inputs.
4. Fill a private copy of `config/cluster.example.json` under `local/`. Render each rank's command with `scripts/render_launch.py`; review and run it on the corresponding machine only after preparation and drain checks.
5. Keep the raw head listener on loopback. Supply your own authenticated gateway before exposing an API. No private gateway, credentials, key policy or cluster network configuration is included here.

The renderer prints commands only. It never stops services, changes networks, adjusts clocks, deletes caches, starts containers or sends model requests.

## Layout

- `overlay/`: 195 changed/new engine files, applied to a pinned public base.
- `provenance.json`: upstream pins, base hashes and deployed/export source hashes.
- `tools/export/`: CPU-only whole-block layout, rank export, source verification and tests.
- `tools/kernels/`: retained throughput kernel sources and offline compiler cases.
- `config/`: portable profile, example host configuration and worker build arguments.
- `tools/benchmark.py`: retained synthetic black-box benchmark client.
- `licenses/`: original donor notices and license texts.

Apache-2.0 for integration code, with upstream components retaining their stated licenses. Weights and third-party runtimes have separate licenses. See [CREDITS.md](CREDITS.md).
