# fedotmas-meta

The meta-agent: synthesizes agent systems over the engine.

A system is proposed as data (`SystemSpec`: a preset name plus a filling of its roles),
checked against the preset it names (`assemble`), and only then built. The provider is
involved in one place, `compose`, which has a model write the filling and sends a batch back
with its problems if it does not fit.

```python
from fedotmas_meta import Catalog, assemble, compose
from fedotmas_meta.presets import SwarmPreset

topic = "Should model weights be open?"
preset = SwarmPreset()
composed = await compose(topic, preset, llm=backend, count=40, batch=10)
board = assemble(composed.spec, Catalog(preset))
await board.run(
    preset.seed(topic, composed.spec.fill["personas"]),
    goal="__never__",
    bind={"llm": backend},
    budget=15,
)
```

`batch` is load-bearing above a dozen or so agents: asked for forty voices in one answer a
model quietly returns thirty-something, every time. Asked for ten at a time, told who is in
the room already, it fills the room exactly. `fallback=` takes a handwritten spec to stand in
when it cannot, so a composed run never fails for want of a cast.

`Preset` is a protocol, not a menu: a pattern family with named role slots and a `build` that
turns a filling into anything the engine can compile. `SwarmPreset` is the one shipped here,
the shape `benchmarks/swarm` measures; pass `ranker=by_interest` to give each persona its own
feed instead of one wall everybody reads.
