# Secure engine source and deployment boundary

The overlay includes the mixed-serving secure listener, request isolation, all-rank cleanup, graph/private-allocation handling and scoped memory-only prefix-cache implementation used by the retained deployment. This exports the engine code, **not a complete hardened deployment or a transferable secure-chat qualification**. The public launch profile does not set secure-listener or secure-cache flags.

The implemented protocol uses a separately authenticated listener, `X-Secure-Cache-Scope` with a random 64-character lowercase hexadecimal scope, and zero-body `POST /api/v1/secure/forget`. A scope header forgets only that namespace; no header forgets all secure namespaces. Missing scopes retain no-reuse behavior. The configured deployment policy was eight entries, 2 GiB per rank, 15-minute idle expiry, with error and explicit-forget cleanup. Cache hits use `usage.prompt_tokens_details.cached_tokens`. Ordinary requests must not access secure namespaces. Secure image requests remain refused.

The native listener refuses to start without the expected preloaded runtime guard and verified all-rank reset support. That check alone is not a complete privacy boundary. The private deployment's supervisor, credential management, guard build, custom collective-library cleanup, per-process no-swap controls and host policies are not packaged here. There are no deployed addresses, keys, key digests or user-chat records in this repository.

A downstream integration must supply and review those protections, prove cleanup on every rank and all terminal paths, prevent disk spill/logging/dumps/swap of secure state, and test scoped reuse, isolation, expiry, cancellation/error and forget races. It must also qualify the exact authenticated client route and measure ordinary-only and mixed workloads. Do not bypass the guard or label a raw listener secure because the API responds.

Published timings use synthetic inputs only. They describe the retained deployment, not an installation of this public package. The source tests are useful regression coverage and do not establish a new machine's privacy or performance.
