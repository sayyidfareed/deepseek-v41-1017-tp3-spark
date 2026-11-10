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

## Coding quality

The Nov 5 combined throughput build scored **156/164 HumanEval** and **147/164 HumanEvalPlus-Mini**. Its evaluation reused 101 saved generations and completed 63 remaining tasks. This was not a fresh 164-task generation run of the Nov 9 refresh.
