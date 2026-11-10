# Measurements and their scope

`2026-11-09.json` is an allowlisted numeric summary from the deployed source refresh. No request bodies, private traces, endpoints or conversation cache are published.

The retained client is adapted from BertholomusAI's kernel-kit benchmark; credit and source lineage remain in its header and CREDITS. Decode speed excludes the first token: `(completion_tokens - 1) / (last_chunk_time - first_chunk_time)`. Concurrent aggregate throughput uses the runner's sustained-window accounting; it is not the mean per-request speed.

The measured commands used the equivalent of:

```sh
python3 tools/benchmark.py --base http://127.0.0.1:8000 --model deepseek-v4.1-flash-1017-tp3 decode --set b --tokens 384 --reps 1 --out local/c1.json
python3 tools/benchmark.py --base http://127.0.0.1:8000 --model deepseek-v4.1-flash-1017-tp3 sustained --streams 4,8 --seconds 20 --set b --tokens 384 --out local/c4-c8.json
```

Check the runner's `--help` for exact options. These commands send synthetic inference requests and consume the shared cluster; run only during an agreed idle window. They are not executed by source preparation or CI. The current output-cap, prompt mix, temperature/thinking mode, cache condition, clock cap and all active contenders must be recorded for a fair comparison.

The source refresh passed 697 CPU tests in the deployment build. Historical November 5 coding scores are 156/164 HumanEval and 147/164 HumanEvalPlus-Mini; the combined throughput continuation reused 101 already-saved tasks and generated 63. They are not 164 fresh generations of the November 9 source, nor proof about hosted model precision. Speed/quality conclusions must not cross those protocols silently.
