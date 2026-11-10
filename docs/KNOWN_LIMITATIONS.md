# Known limitations

- **Slowdowns:** bounded text/image prefill scheduling and decode-service fixes are included and passed synthetic mixed-workload checks. Intermittent low throughput and long first-token delays still occur; these fixes do not establish sustained mixed-workload performance or explain every pause. The 250 ms allowance budgets decode service, not a maximum prefill or SSE pause.
- **Long contexts:** twelve request slots do not guarantee twelve simultaneous long histories. Near-1M context operation is not qualified.
- **Reasoning:** malformed or unmarked thinking blocks can still leak into answer text. Incremental streaming remains enabled.
- **JSON schemas:** root or nested `anyOf`/`oneOf` object unions return HTTP 422. JSON-object fallback requires client-side validation.
- **Caching:** TP2 and TP3 spill caches are not interchangeable. In-memory cache is lost on shutdown; no cross-topology migration is provided.
- **Retention:** ordinary disk spill retains conversation-derived state. No secure/no-retention qualification is claimed.
- **Installation:** source reconstruction is tested; fresh-machine runtime assembly and deployment of this public packaging are not. There is no bundled image, automatic network setup or service supervisor.
