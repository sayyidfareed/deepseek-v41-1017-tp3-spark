# Credits and source lineage

This recipe integrates the following upstream work. Attribution does not imply endorsement.

| Contributor / project | Work used here |
|---|---|
| [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | Original V4.1 Flash 1017 model, architecture, tokenizer and Engram weights. |
| [Ash Hart and TensorFold contributors](https://github.com/ashhart/TensorFold) | Native Zig serving engine, CUDA/lanes infrastructure, EXL3 support and the upstream kernel implementations used by the donors. |
| [Albert Lee / BertholomusAI](https://github.com/bertholomus/TensorFold) | DeepSeek V4.1 native model implementation, TP execution, vision, Engram, draft support, input tools and kernel kit. Base engine pinned in `provenance.json`. |
| [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks) | 1017 EAQ v1 2.9-bpw quantization, TP2 recipe and cache/scheduling/API corrections carried into TP3. Quantized weights remain separately distributed by Mia. |
| [LakyFx](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-tp4-spark) | Three-rank geometry, physical padding/logical-vocabulary handling, NCCL integration, smart graphs, adaptive speculation, prefill and mid-batch throughput donor code at the pinned revision. We use Mia's 1017 weights, not the older donor weight export. |
| [Turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3) | EXL3 format and decoding/kernel lineage. MIT notices retained. |
| [Robert Wojciechowski / wojo](https://github.com/wojo) and [BadAd84](https://github.com/BadAd84) | Disk-spill/cache design and fixes credited by Mia, including block-tier design and CPU validation. |
| [eleata](https://github.com/eleata) | Mia's piecewise-prefill engine rebase and associated integration. |
| [Christopher Owen / spark-ds41f](https://github.com/christopherowen/spark-ds41f) | Reference for the mixed-prefill scheduling comparison that informed our fairness repair. No Owen runtime, kernels or performance results are included as our own. |
| [Hector Isaza / hecisaza](https://github.com/hecisaza) | Upstream Engram failure-path and server/parser work, including adopted 0064/0065 DSML fixes. |
| [Christopher A. Williamson / cawilliamson](https://github.com/cawilliamson) | GET/DELETE request-body draining correction carried by Mia as 0046. |
| [Lukasz Raczylo](https://github.com/lukaszraczylo) | First-draft-depth correction 0049 carried by Mia. Optional 0050/0051 features were not activated by this integration. |
| [Mykhailo Tamarin](https://github.com/MykhailoTamarin), [Coinupbtc](https://github.com/Coinupbtc), [hashd1ve](https://github.com/hashd1ve) | Mixed-pack/loader compatibility, earlier recipe patches, and template construction contributions credited in the retained Mia notices. Our selected weights remain EAQ v1. |
| NVIDIA, PyTorch, Triton, Pillow and other upstream dependencies | CUDA/NCCL, attention and kernel/runtime/image components; each retains its own license. No proprietary runtime or binary redistribution is included. |

## This integration

Our contribution is combining Mia's exact 1017 EAQ pack with the three-rank donor runtime, lossless whole-block rank export, conflict resolution preserving TP3 behavior, selected TP2 correctness updates, local grammar/reopened-thinking fixes, bounded text/image prefill scheduling with decode service and regression coverage, deployment configuration and bounded qualification. The public repository is maintained by [sayyidfareed](https://github.com/sayyidfareed).

See `provenance.json` for pinned repositories and the base/deployed/export SHA256 of each changed engine file. Exported source adds modification notices; those comments are the only intended difference from the captured deployed source. JSON fixtures remain byte-identical where no comment is possible.

## Original notices, retained in full

- [TensorFold/Bertholomus NOTICE](licenses/tensorfold/NOTICE), [third-party notices](licenses/tensorfold/THIRD_PARTY_NOTICES.md), [DeepSeek attribution](licenses/tensorfold/DSV41-ATTRIBUTION.md).
- [Mia NOTICE](licenses/mia/NOTICE) and [Mia contributor credits](licenses/mia/CREDITS.md). These are the donor's original documents; their references to files, optional features and additional contributors describe that donor, not a claim that every feature is included here.
- [Laky donor NOTICE](licenses/laky/NOTICE), [third-party notices](licenses/laky/THIRD_PARTY_NOTICES.md), [kernel lineage](licenses/kernels/NOTICE.md).
- Full Apache, MIT, BSD and image-component license texts are retained under `licenses/`.

Code is distributed under Apache-2.0 except components retaining their stated upstream licenses. Model weights retain their separate model licenses and are not bundled. Copyright and attribution are not replaced by this repository's maintainer name.
