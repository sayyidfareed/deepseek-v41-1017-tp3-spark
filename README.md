# DeepSeek V4.1 Flash 1017 · 3× DGX Spark

DeepSeek 1017 on three 128 GB NVIDIA GB10 systems, using **Mia EAQ v1 EXL3 2.9 bpw** weights and a **TensorFold TP3** runtime. Combines Mia, LakyFx and BertholomusAI work without requantizing the weights.

**12 request slots · 6.4M shared KV pool · 65,536 max output tokens**

## Benchmarks

| Test | Result |
|---|---:|
| Single request — code | **101 tok/s** |
| Single request — prose | **66 tok/s** |
| Single request — structured text | **147 tok/s** |
| 4 concurrent requests | **154 tok/s aggregate** |
| 8 concurrent requests | **201 tok/s aggregate** |
| 12 concurrent requests | **222 tok/s aggregate** |
| C12 median request decode speed | **23.4 tok/s** |
| C12 time to first token, median / p95 | **0.31 / 2.36 seconds** |
| HumanEval | **95.1% · 156/164** |
| HumanEvalPlus-Mini | **89.6% · 147/164** |

Speed tests use short synthetic prompts, thinking off and a 384-token output cap. Concurrent throughput is the total across requests. C12 and coding results predate the Nov 9 refresh; all results above predate the current scheduling fixes below. They are not a matched scaling comparison or a guarantee for long contexts. [Protocol and results →](benchmarks/README.md)

## Streaming and cache improvements

The current overlay reuses completed Engram reads, bounds read-ahead to the active chunk, and yields while disk-cache restores are pending. Competing prefill uses **1,024-row chunks** and waits for GPU completion before giving decode its 250 ms service allowance; solo prefill retains its 4,096-row maximum. Streaming and cancellation remain enabled.

Measured on the 64 KiB host profile, with unchanged Mia weights, C12 and the 6.4M pool:

| Matched 320K cold-image test + 3 streaming peers | Before 1K chunks | Current |
|---|---:|---:|
| Peer visible output during cold prefill | 7.03 tok/s | **38.83 tok/s** |
| Longest peer output gap | 1.12 s | **0.82 s** |
| Cold image time to first output | **159 s** | 244 s |
| Separate C4 throughput check | 156.57 tok/s | 159.01 tok/s |

This prioritizes ongoing replies at the cost of slower cold-prefill completion. It is one matched pair, not a general throughput claim. After a normal same-build restart, two 32K image histories each restored **30,720 tokens from disk**: first output in **1.06 s** alone and **1.46 s** alongside three streaming peers, whose longest gap was **0.46 s**. [Protocol, results and limits →](benchmarks/README.md#current-streaming-and-restart-cache-checks)

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

Built on **DeepSeek**, **Ash Hart / TensorFold**, **Albert Lee / BertholomusAI**, **MiaAI-Lab**, **LakyFx**, **Turboderp / ExLlamaV3**, **[Christopher Owen](https://github.com/christopherowen)** (Memory Saver and host/scheduling reference), and the cache, kernel and parser contributors credited in [CREDITS.md](CREDITS.md).

Integration code is Apache-2.0; upstream code and model weights retain their own licenses. Original notices are preserved in [licenses/](licenses/).
