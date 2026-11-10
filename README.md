# DeepSeek V4.1 Flash 1017 · 3× DGX Spark

DeepSeek 1017 on three 128 GB NVIDIA GB10 systems, using **Mia EAQ v1 EXL3 2.9 bpw** weights and a **TensorFold TP3** runtime. Combines Mia, LakyFx and BertholomusAI work without requantizing the weights.

**12 request slots · 6.4M shared KV pool · 65,536 max output tokens**

## Benchmarks

| Test | Result | Measurement |
|---|---:|---|
| Single request — code | **101 tok/s** | Nov 9 refresh |
| Single request — prose | **66 tok/s** | Nov 9 refresh |
| Single request — structured text | **147 tok/s** | Nov 9 refresh |
| 4 concurrent requests | **154 tok/s aggregate** | Nov 9 refresh, 20 seconds |
| 8 concurrent requests | **201 tok/s aggregate** | Nov 9 refresh, 20 seconds |
| 12 concurrent requests | **222 tok/s aggregate** | Earlier TP3 build, Nov 5, 90 seconds |
| C12 median request decode speed | **23.4 tok/s** | Same C12 run; 58 requests completed |
| C12 time to first token, median / p95 | **0.31 / 2.36 seconds** | Same C12 run |
| HumanEval | **95.1% · 156/164** | Earlier TP3 build, Nov 5 |
| HumanEvalPlus-Mini | **89.6% · 147/164** | Earlier TP3 build, Nov 5 |

Speed tests use short synthetic prompts, thinking off and a 384-token output cap. Concurrent throughput is the total across requests. C12 and coding results predate the Nov 9 refresh; they are not a matched scaling comparison or a guarantee for long contexts. [Protocol and raw results →](benchmarks/README.md)

## Setup

This is a **source recipe**, not a ready-to-run image or one-command installer. Model weights and binaries are separate; a fresh-machine installation of this packaging has not been tested.

1. [Prepare the source, weights and kernel kit](docs/REPRODUCING.md).
2. Fill in a private copy of [`config/cluster.example.json`](config/cluster.example.json).
3. Render the three rank commands with `scripts/render_launch.py`, then run them on your prepared cluster.
4. Put an authenticated gateway in front of the loopback-only head listener.

[`config/profile.json`](config/profile.json) contains the runtime settings. Source revisions and checksums are pinned in [`provenance.json`](provenance.json).

## Features and limits

- Whole-block TP3 weight export, GPU candidate sampling, adaptive speculation, prefill and mid-batch EXL3 optimizations.
- Streaming, tools, image input, reasoning controls and disk-backed prompt/KV caching.
- The 6.4M pool is shared; it is not a per-request context limit. Twelve long histories may not all fit concurrently. KV sharding is off.
- Intermittent slowdowns and malformed reasoning-tag leaks remain unresolved. Object-union JSON schemas return HTTP 422.
- Disk caching retains conversation-derived state; this recipe is not suitable for no-retention chats.

[Known limitations →](docs/KNOWN_LIMITATIONS.md)

## Credits

Built on **DeepSeek**, **Ash Hart / TensorFold**, **Albert Lee / BertholomusAI**, **MiaAI-Lab**, **LakyFx**, **Turboderp / ExLlamaV3**, and the cache, kernel and parser contributors credited in [CREDITS.md](CREDITS.md).

Integration code is Apache-2.0; upstream code and model weights retain their own licenses. Original notices are preserved in [licenses/](licenses/).
