"""The swarm of `packages/fedotmas-llm/tests/test_swarm.py` on a real provider instead of a
stub: N personas argue a topic on a shared feed for R rounds, with ActivitySample deciding who
speaks each round, ConcurrencyLimit capping in-flight requests and SqliteStore holding the feed,
and the provider's token usage reported for the whole run.

Costs money and needs OPENROUTER_API_KEY in .env, so it lives here and not under pytest.

Usage:
uv run --group examples python benchmarks/swarm/run.py --personas 2 --rounds 1 --concurrency 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fedotmas import Plugin, Rule, blackboard
from fedotmas.engine import ActivitySample, PluginDispatcher, SqliteStore, View
from fedotmas.engine.contract import Fact, Node, Result
from fedotmas.ext.plugins import ConcurrencyLimit, Retry
from fedotmas_llm import PromptRule
from fedotmas_llm.adapters.pydantic_ai import PydanticAI
from pydantic_ai.exceptions import ModelHTTPError

TOPIC = "Should frontier AI labs release model weights openly?"

VOICES = [
    ("an open-source maintainer", "argues from what shipping in public actually costs"),
    ("a safety researcher", "keeps pulling the thread back to misuse"),
    ("a startup founder", "cares about who can afford to compete"),
    ("an academic", "wants reproducibility above everything"),
    ("a policy advisor", "thinks in terms of what is enforceable"),
    ("a security engineer", "asks who patches it after release"),
    ("a journalist", "pushes for a plain answer, distrusts jargon"),
    ("an economist", "reduces the argument to incentives"),
]

FEED_WIDTH = 12


class Watch(Plugin):
    """Innermost plugin: sits under Retry so it counts every attempt (attempts minus fired
    rules is the retry count), under ConcurrencyLimit so its peak is real in-flight requests,
    and records what the provider actually raised."""

    def __init__(self) -> None:
        self.attempts = 0
        self.peak = 0
        self.failures: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self._live = 0

    async def around_node(self, node, input, view, call) -> Result:
        self.attempts += 1
        self._live += 1
        self.peak = max(self.peak, self._live)
        try:
            return await call(input, view)
        except Exception as e:
            self.failures[type(e).__name__] += 1
            raise
        finally:
            self._live -= 1

    async def on_error(self, node: Node, error: Fact, view: View) -> None:
        self.errors[node.describe().name] += 1


def _clock() -> Rule:
    async def tick(v: int) -> int:
        return v + 1

    return Rule(name="clock", fn=tick, reads="tick", writes="tick", when=lambda v: True)


def _feed() -> Rule:
    """Infrastructure (no activity_level, so ActivitySample always fires it): fold the posts
    so far into the blurb the personas read next round. Writes commit at the end of a
    superstep, so the feed a persona sees trails the live store by one round."""

    async def digest(_: int, view: View) -> str:
        posts = view.query("post")[-FEED_WIDTH:]
        return "\n".join(f"{f.producer}: {f.value}" for f in posts) or "(nothing yet)"

    return Rule(
        name="feed", fn=digest, reads="tick", writes="feed", when=lambda v: True
    )


def _personas(n: int, rng: random.Random) -> list[PromptRule]:
    """One rule per persona, each with its own activity_level: the fedotmas analogue of the
    per-agent activity profile an OASIS-style simulation samples its speakers from."""
    rules = []
    for i in range(n):
        role, quirk = VOICES[i % len(VOICES)]
        rules.append(
            PromptRule(
                name=f"persona_{i}",
                prompt=(
                    f"You are persona {i}, {role} who {quirk}. You post on a public feed. "
                    "Write ONE post of at most two sentences, in character, answering the "
                    "topic and reacting to the feed if it is not empty. No preamble, no "
                    "quotes, no hashtags."
                ),
                input="Topic: {topic}\nRound: {input}\nFeed so far:\n{feed}",
                reads="tick",
                writes="post",
                when=lambda v: True,
                meta={"activity_level": rng.uniform(0.2, 1.0)},
            )
        )
    return rules


async def main(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    settings: dict[str, Any] = {"max_tokens": args.max_tokens, "temperature": 0.9}
    if args.no_reasoning:
        settings["extra_body"] = {"reasoning": {"enabled": False}}
    backend = PydanticAI(model=args.model, model_settings=settings)
    watch = Watch()
    plugins = PluginDispatcher(
        # only HTTP failures are worth another attempt: a 429 or a 5xx clears, a bad
        # prompt or a blown token budget just repeats
        [
            Retry(args.retries, on=ModelHTTPError),
            ConcurrencyLimit(args.concurrency),
            watch,
        ]
    )
    board = blackboard(
        _clock(),
        _feed(),
        *_personas(args.personas, rng),
        policy=ActivitySample(args.min_active, args.max_active, rng=rng),
        halt_on_error=False,
    )
    system = board.system(bind={"llm": backend}, plugins=plugins)
    store = SqliteStore(args.db)

    started = time.monotonic()
    run = await system.run(
        {"tick": 0, "topic": args.topic, "feed": "(nothing yet)"},
        goal="__never__",  # nothing writes this tag: only the round budget ends the run
        budget=args.rounds,
        plugins=plugins,
        store=store,
    )
    elapsed = time.monotonic() - started

    posts = store.snapshot().query("post")
    store.close()

    usage = backend.usage
    fired = sum(1 for s in run.steps for name in s.fired if name.startswith("persona_"))
    invocations = sum(len(s.fired) for s in run.steps)
    cost = (
        usage.input_tokens * args.price_in + usage.output_tokens * args.price_out
    ) / 1e6
    return {
        "model": args.model,
        "personas": args.personas,
        "rounds": len(run.steps),
        "reason": run.reason,
        "active_per_round": [
            sum(1 for name in s.fired if name.startswith("persona_")) for s in run.steps
        ],
        "personas_fired": fired,
        "posts": len(posts),
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        # reasoning is billed inside output_tokens; fedotmas_llm.Usage does not model the
        # split, so read the provider's own breakdown off the adapter's RunUsage
        "reasoning_tokens": backend._usage.details.get("reasoning_tokens", 0),
        "attempts": watch.attempts,
        "retried": watch.attempts - invocations,
        "raised": dict(watch.failures),
        "failed_nodes": dict(watch.errors),
        "peak_in_flight": watch.peak,
        "seconds": round(elapsed, 1),
        "usd": round(cost, 6),
        "sample": [f"{f.producer}: {f.value}" for f in posts[:3]],
    }


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openrouter:qwen/qwen3.8-flash")
    p.add_argument("--personas", type=int, default=10)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--min-active", type=int, default=2)
    p.add_argument("--max-active", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--no-reasoning", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--topic", default=TOPIC)
    p.add_argument("--db", default=str(Path(__file__).parents[1] / "out" / "swarm.db"))
    p.add_argument(
        "--report", default="", help="where to write the report (default: <db>.json)"
    )
    p.add_argument(
        "--price-in", type=float, default=0.15, help="USD per 1M input tokens"
    )
    p.add_argument("--price-out", type=float, default=0.47, help="USD per 1M output")
    return p.parse_args()


if __name__ == "__main__":
    args = cli()
    load_dotenv(Path(__file__).parents[2] / ".env")
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    Path(args.db).unlink(missing_ok=True)
    report = json.dumps(asyncio.run(main(args)), indent=2, ensure_ascii=False)
    Path(args.report or f"{args.db}.json").write_text(report)
    print(report)
