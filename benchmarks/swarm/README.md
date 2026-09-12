# Swarm

What a swarm of prompt rules costs on a real provider. The system is the one in
`packages/fedotmas-llm/tests/test_swarm.py` with the stub LLM swapped for `PydanticAI`:
N personas with per-agent `activity_level` argue a topic on a shared feed, `ActivitySample`
picks who speaks each round, `ConcurrencyLimit` caps in-flight requests, `Retry` covers
provider HTTP failures, and `SqliteStore` holds the feed in a file. The stub test stays the
deterministic unit test; this is the paid, non-deterministic half, so it lives outside pytest.

```bash
uv sync --all-packages --extra pydantic-ai
echo 'OPENROUTER_API_KEY=...' >> .env
uv run --group examples python benchmarks/swarm/run.py --personas 40 --rounds 15
```

A run writes its report to `benchmarks/out/<db>.json` and leaves the feed itself in the
SQLite file beside it, readable mid-run from another process.

## Results

`qwen/qwen3.8-flash` via OpenRouter, 2026-09-12. 40 personas, 15 rounds,
`ActivitySample(3, 8)`, `ConcurrencyLimit(3)`, `Retry(3, on=ModelHTTPError)`,
`max_tokens=800`, `temperature=0.9`, seed 7. Cost at the catalog price of $0.15/M input and
$0.47/M output.

| run | personas x rounds | requests | input | output | of which reasoning | USD | wall |
|---|---|---|---|---|---|---|---|
| smoke | 2 x 1 | 1 | 101 | 293 | 257 | 0.0002 | 17s |
| small | 10 x 3 | 10 | 1 832 | 2 959 | 2 439 | 0.0017 | 40s |
| small, no reasoning | 10 x 3 | 10 | 1 544 | 496 | 0 | 0.0005 | 22s |
| full | 40 x 15 | 76 | 56 693 | 29 914 | 26 428 | 0.0226 | 262s |
| full, no reasoning | 40 x 15 | 76 | 51 745 | 4 199 | 0 | 0.0097 | 92s |

Reading the full run: the throttle turned 600 possible persona-rounds into 76 requests
(activation per round 3..8, mean 5.1), and the concurrency cap held at 3 in flight
throughout. Per request that is ~750 input and ~390 output tokens, input growing with the
feed and flat past the 12-post window the digest keeps.

Three things the numbers say:

- **Reasoning dominates the bill.** 88% of output tokens on this model are hidden reasoning
  for a two-sentence post. `--no-reasoning` sends OpenRouter's `reasoning: {enabled: false}`
  and cuts cost 2.3x and wall time 2.9x with no visible loss on this task.
- **Retry never fired.** Zero 429s and zero 5xx across the 173 requests at concurrency 3, so the
  cap is conservative for this key rather than load-bearing. The two failures in the full run
  were `UnexpectedModelBehavior`: reasoning consumed the whole `max_tokens` budget before any
  output, which no retry can fix, hence `on=ModelHTTPError` rather than `on=Exception`. With
  reasoning off, zero failures.
- **The swarm is cheap.** 2.3 cents for a 40-agent, 15-round conversation; the run is bounded
  by latency, not by budget.

The feed a persona reads trails the store by one round: a superstep commits at its end, so
the digest written in round n is read in round n+1 and covers posts through round n-1.
