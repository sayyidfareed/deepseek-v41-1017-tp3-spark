# Benchmark protocol

| Result file | Build | Workload |
|---|---|---|
| [2026-11-09.json](2026-11-09.json) | Nov 9 TP3 refresh | C1: one run per prompt; C4/C8: 20-second windows; GPU ceiling 2,200 MHz |
| [2026-11-05-c12.json](2026-11-05-c12.json) | Earlier Mia TP3 | C12: 90-second window, 58 completed requests |

Speed runs use prompt set B, temperature 0, thinking off and a 384-token output cap. C12 was not rerun on the Nov 9 refresh; its clock setting is not established by the retained summary. These are short-prompt measurements, not long-history capacity tests or a matched TP2/TP3 comparison.

Per-request decode speed excludes the first token: `(completion_tokens - 1) / (last_chunk_time - first_chunk_time)`. Aggregate throughput uses the runner's sustained-window accounting; it is not the mean per-request speed.

## Run the speed tests

The client is adapted from BertholomusAI's kernel-kit benchmark. Run against an idle cluster:

```sh
python3 tools/benchmark.py --base http://127.0.0.1:8000 --model deepseek-v4.1-flash-1017-tp3 decode --set b --tokens 384 --reps 1 --out local/c1.json
python3 tools/benchmark.py --base http://127.0.0.1:8000 --model deepseek-v4.1-flash-1017-tp3 sustained --streams 4,8,12 --seconds 90 --set b --tokens 384 --out local/concurrent.json
```

Record the build, clocks, cache state and competing work with each run. The example uses 90 seconds at every concurrency; the Nov 9 C4/C8 results used 20 seconds.

## Earlier mixed prefill and streaming

The scheduling repairs use the same weights, kernels, C12 capacity, 6.4M pool and 2,200 MHz ceiling. Text checks used the text repair; the image check used the subsequent combined text/image repair. One synthetic run per case; there is no matched incumbent GPU control and no new C1/C4/C8/C12 throughput score for these repairs.

| Workload | Warm median TTFT | Longest warm SSE gap | Cold TTFT |
|---|---:|---:|---:|
| Cold 131K text + 3 warm 32K requests | 4.05 s | 1.14 s | 67.80 s |
| Cold 131K text + 7 warm 32K requests | 6.18 s | 1.21 s | 76.10 s |

Cold text requests reported zero cached tokens; each warm request reused 30,720 tokens. All warm replies started before the cold reply. Serving those replies can lengthen cold-prefill completion; no isolated-prefill improvement is claimed.

For the combined repair, a 32,231-token image-bearing prompt with zero cached tokens ran alongside three already-streaming peers. Each peer delivered 21 SSE chunks during its prefill; the longest delivery gap was 1.684 s. The image answer was correct with 20.307 s TTFT. Repeating that image history reused 30,720 tokens with 0.717 s TTFT. A synthetic disconnect drained in 3.01 s; a subsequent different-color image answered correctly in 2.15 s. SSE gaps measure delivery intervals, not per-token GPU latency.

The retained CPU suite passes 700 tests (497 executed on the combined repair, 203 unchanged cached successes), including peer progress, cancellation and image-state exclusivity. The exported five changed source files match the tested source except for their existing attribution comments. This publication adds no new GPU run and does not claim every ordinary-workload slowdown is fixed.

## Coding quality

The Nov 5 combined throughput build scored **156/164 HumanEval** and **147/164 HumanEvalPlus-Mini**. Its evaluation reused 101 saved generations and completed 63 remaining tasks. This was not a fresh 164-task generation run of the Nov 9 refresh.

## Current streaming and restart-cache checks

The Nov 10 overlay was tested on three GB10s with kernel `7.0.0-1019-nvidia-64k`, NVIDIA `580.178.04`, Memory Saver `0.4.1` with UVM packing, and a 2,200 MHz GPU ceiling. Exact Mia EAQ 2.9-bpw weights, streaming, 12 slots and pool 6.4M were retained. These measurements do not isolate a kernel or Memory Saver speedup.

[Machine-readable results](2026-11-10-streaming-cache.json) contain only synthetic workload aggregates.

The matched prefill comparison used one cold 320,231-token image history with zero cached tokens alongside three streaming peers per build. Moving from 2,048-row competing chunks to 1,024 rows (with the completion fence) raised peer output during prefill from 7.03 to 38.83 visible tok/s (retokenized visible output, not sampled-token telemetry) and reduced the longest peer gap from 1.12 to 0.82 seconds. Cold first output increased from 159.14 to 244.35 seconds; whole-wave time increased from 286.22 to 296.61 seconds. Separate C4 checks were 156.57 and 159.01 aggregate tok/s; that small difference is not a noise-resolved speedup. The images were answered correctly.

A separate normal-to-normal same-build restart check seeded two fresh synthetic 32,231-token image histories and verified durable writes on all three ranks. Both then recorded a committed disk hit and 30,720 restored tokens. Idle TTFT was 1.058 seconds; with three streaming peers it was 1.464 seconds. Each peer completed 2,048 output tokens, progressed before the image's first output, and had a maximum SSE gap no greater than 0.461 seconds. All 18 ordinary API-route checks and 15 key-policy checks passed. No model-process swapping or OOM was observed.

This confirms same-build restart reuse for those two histories. It does not establish cross-build/topology compatibility, uninterrupted service under every mixed workload, or a fix for every historical slowdown. Earlier diagnostic-mode starts missed their checkpoints; normal-mode recovery passed. No private prompts, cache contents, host identities or credentials are included in these results. Coding quality and the short-prompt table above were not rerun for this publication.
