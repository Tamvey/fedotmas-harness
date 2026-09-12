from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable
from typing import Any

from fedotmas import Board, Rule, blackboard
from fedotmas.engine import ActivitySample, View
from fedotmas.engine.contract import Fact
from fedotmas_llm import PromptRule

from fedotmas_meta._assemble import Agent, Fill
from fedotmas_meta._spec import RoleSpec, SpecError

CONDUCT = (
    "You post on a public feed. Write ONE post of at most two sentences, in character, "
    "answering the topic and reacting to the feed if it is not empty. No preamble, no "
    "quotes, no hashtags."
)
FEED = "Topic: {topic}\nRound: {input}\nFeed so far:\n{feed}"
EMPTY = "(nothing yet)"

Ranker = Callable[[Agent, list[Fact]], list[Fact]]


def by_interest(agent: Agent, posts: list[Fact]) -> list[Fact]:
    """Stand-in for the recommender OASIS puts between the platform and an agent: order posts
    by how much of their vocabulary the persona already uses, recent first among equals. Word
    overlap rather than embeddings, so it stays free and offline."""
    words = _words(agent.prompt)
    scored = [(len(words & _words(str(f.value))), i, f) for i, f in enumerate(posts)]
    return [f for _, _, f in sorted(scored, key=lambda s: s[:2], reverse=True)]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


class SwarmPreset:
    """Many personas posting to one shared feed over a run of rounds: the clock re-arms every
    persona each round, ActivitySample decides which of them actually speak, and a feed rule
    folds the posts so far into what they read next. A spec fills the voices; cadence and feed
    width are the preset's own, so a composed swarm stays comparable to a handwritten one.

    With `ranker` set, each persona gets its own feed rule and reads its own ranking of the
    same posts, which is what lets a room split rather than converge. Seed a run with
    `seed(topic)` and let it end on its round budget: nothing writes a goal.
    """

    name = "swarm"
    hint = "many personas argue a topic on one shared feed, a few speaking each round"
    roles = (
        RoleSpec(
            "personas", "one per voice you want in the room, keyed by a short id", True
        ),
    )
    reserved = frozenset({"clock", "feed"})

    def __init__(
        self,
        *,
        feed_width: int = 12,
        min_active: int = 3,
        max_active: int | None = 8,
        activity: tuple[float, float] = (0.2, 1.0),
        ranker: Ranker | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.feed_width = feed_width
        self.min_active = min_active
        self.max_active = max_active
        self.activity = activity
        self.ranker = ranker
        self._rng = rng or random.Random()

    def seed(self, topic: str, personas: Iterable[str] = ()) -> dict[str, Any]:
        """The opening facts. With a ranker every persona reads its own feed tag, and a tag
        nothing has written yet is a missing template reference, so the cast has to be named
        here for the first round to run at all."""
        feeds = ["feed"] if self.ranker is None else [f"feed_{n}" for n in personas]
        return {"tick": 0, "topic": topic} | {tag: EMPTY for tag in feeds}

    def build(self, fill: Fill) -> Board:
        personas = fill["personas"]
        assert isinstance(personas, dict)
        agents = list(personas.values())
        feeds = [self._feed()] if self.ranker is None else self._ranked(agents)
        return blackboard(
            self._clock(),
            *feeds,
            *(self._persona(a) for a in agents),
            policy=ActivitySample(self.min_active, self.max_active, rng=self._rng),
            halt_on_error=False,
        )

    def _clock(self) -> Rule:
        """Rewrites `tick` every round, which is what re-arms the personas: the executor fires
        a node once per distinct matched input, so a fresh tick is a fresh input for all."""

        async def tick(v: int) -> int:
            return v + 1

        return Rule(
            name="clock", fn=tick, reads="tick", writes="tick", when=lambda v: True
        )

    def _feed(self, agent: Agent | None = None) -> Rule:
        """Infrastructure, so it carries no activity_level and always fires. Writes land at
        the end of a superstep, so the feed a persona reads trails the store by one round."""
        width, ranker = self.feed_width, self.ranker
        tag = _feed_tag(agent)

        async def digest(_: int, view: View) -> str:
            posts = view.query("post")
            if ranker is not None and agent is not None:
                keep = {f.key for f in ranker(agent, posts)[:width]}
                posts = [f for f in posts if f.key in keep]
            return (
                "\n".join(f"{f.producer}: {f.value}" for f in posts[-width:]) or EMPTY
            )

        return Rule(name=tag, fn=digest, reads="tick", writes=tag, when=lambda v: True)

    def _ranked(self, agents: list[Agent]) -> list[Rule]:
        taken = {a.name for a in agents} & {_feed_tag(a) for a in agents}
        if taken:
            raise SpecError(f"personas {sorted(taken)} shadow their own feed rules")
        return [self._feed(a) for a in agents]

    def _persona(self, agent: Agent) -> PromptRule:
        """The spec supplies the character, the preset the conduct: what this medium rewards
        is the preset's business, not the meta-agent's."""
        template = FEED
        if self.ranker is not None:
            template = FEED.replace("{feed}", f"{{{_feed_tag(agent)}}}")
        return PromptRule(
            name=agent.name,
            prompt=f"{agent.prompt}\n{CONDUCT}",
            input=template,
            reads="tick",
            writes="post",
            when=lambda v: True,
            llm=agent.llm,
            tools=list(agent.tools) or None,
            meta={"activity_level": self._rng.uniform(*self.activity)},
        )


def _feed_tag(agent: Agent | None) -> str:
    return "feed" if agent is None else f"feed_{agent.name}"
