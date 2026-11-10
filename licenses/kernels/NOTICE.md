# Attribution

These experiments selectively adapt TensorFold work, pinned at commit
`45fcab6c2fddd88ff2da2e0d86b5123d88796a06`, rather than claiming upstream
algorithms as our invention. Retain original copyright headers.

- TensorFold: https://github.com/ashhart/TensorFold — Apache-2.0,
  `LICENSE.TensorFold`. See PRs 1467 and 1534 for the new EXL3 mechanisms.
- EXL3 decoding derives from ExLlamaV3, Copyright (c) 2025 Turboderp,
  MIT, `LICENSE.ExLlamaV3`, with DeepSeek integration credited in donor headers.
- Unmodified `donor/upstream/` and `donor/reference/` files and their SHA-256
  hashes are preserved in `PROVENANCE.json`.

Our adaptation adds standalone build wrappers, matched cache-policy controls,
old-engine ABI/dispatch integration, preservation of split-K L2 discard,
device-capacity fallback and qualification tooling. These changes do not transfer
authorship of the original kernels. Reference donors come from our retained
`experiments/flint_full_20261022/vendor/tensorfold_kernel_reference` snapshot;
their exact file hashes, rather than an invented upstream revision, identify it.
