"""The swarm preset has to produce the same board the benchmark used to build by hand: the two
infrastructure rules plus one node per persona, the conduct appended to each character, and the
feed rendered into every prompt."""

import random
from typing import Any

import pytest
from fedotmas import Board
from fedotmas.engine import View
from fedotmas_llm import Call, PromptRule
from fedotmas_meta import AgentSpec, Catalog, SpecError, SystemSpec, assemble
from fedotmas_meta.presets import SwarmPreset, by_interest
from fedotmas_meta.presets._swarm import CONDUCT

TOPIC = "should weights be open?"
CAST = ["hawk", "dove", "owl", "crow"]


class StubLLM:
    """A deterministic stand-in that records what each persona was actually asked."""

    def __init__(self) -> None:
        self.calls: list[Call] = []

    async def complete(self, call: Call, view: View) -> Any:
        self.calls.append(call)
        return f"post {len(self.calls)}"


def _spec(names=CAST) -> SystemSpec:
    return SystemSpec(
        preset="swarm",
        fill={"personas": {n: AgentSpec(prompt=f"You are the {n}.") for n in names}},
    )


def _preset(**kw) -> SwarmPreset:
    return SwarmPreset(rng=random.Random(7), **kw)


def _board(spec: SystemSpec, preset: SwarmPreset) -> Board:
    board = assemble(spec, Catalog(preset))
    assert isinstance(board, Board)
    return board


def test_the_board_is_the_personas_plus_the_two_infrastructure_rules():
    preset = _preset()
    board = _board(_spec(), preset)
    assert [r.name for r in board.rules] == ["clock", "feed", *CAST]
    assert not board.halt_on_error


def test_only_the_personas_carry_an_activity_level():
    board = _board(_spec(), _preset(activity=(0.5, 0.5)))
    by_name = {r.name: r.meta for r in board.rules}
    assert by_name["clock"] == by_name["feed"] == {}
    assert all(by_name[n] == {"activity_level": 0.5} for n in CAST)


def test_the_spec_supplies_the_character_and_the_preset_the_conduct():
    board = _board(_spec(), _preset())
    hawk = next(r for r in board.rules if r.name == "hawk")
    assert isinstance(hawk, PromptRule)
    assert hawk.prompt == f"You are the hawk.\n{CONDUCT}"


async def test_a_run_posts_to_the_feed_and_shows_it_back_to_the_personas():
    preset = _preset(min_active=0, max_active=None, activity=(1.0, 1.0))
    board = _board(_spec(), preset)
    stub = StubLLM()

    run = await board.run(
        preset.seed(TOPIC), goal="__never__", bind={"llm": stub}, budget=3
    )

    assert run.reason == "budget"
    assert len(stub.calls) == len(CAST) * 3
    assert run.view.count("post") == len(CAST) * 3

    first, later = stub.calls[0].input, stub.calls[-1].input
    assert f"Topic: {TOPIC}" in first
    assert "(nothing yet)" in first
    assert "hawk: post" in later


async def test_the_feed_a_persona_reads_is_capped_at_the_preset_width():
    preset = _preset(feed_width=2, min_active=0, max_active=None, activity=(1.0, 1.0))
    board = _board(_spec(), preset)
    stub = StubLLM()

    await board.run(preset.seed(TOPIC), goal="__never__", bind={"llm": stub}, budget=3)

    feed = stub.calls[-1].input.split("Feed so far:\n")[1]
    assert len(feed.splitlines()) == 2


def test_a_ranker_gives_every_persona_its_own_feed_rule():
    preset = _preset(ranker=by_interest)
    board = _board(_spec(), preset)
    assert [r.name for r in board.rules] == [
        "clock",
        *(f"feed_{n}" for n in CAST),
        *CAST,
    ]
    hawk = next(r for r in board.rules if r.name == "hawk")
    assert isinstance(hawk, PromptRule)
    assert hawk.input is not None
    assert "{feed_hawk}" in hawk.input


def test_a_persona_that_shadows_its_own_feed_rule_is_refused():
    spec = _spec(["feed_hawk", "hawk"])
    with pytest.raises(SpecError, match="shadow their own feed rules"):
        _board(spec, _preset(ranker=by_interest))


async def test_each_persona_reads_its_own_ranking_of_the_same_posts():
    preset = _preset(
        feed_width=1,
        min_active=0,
        max_active=None,
        activity=(1.0, 1.0),
        ranker=by_interest,
    )
    spec = SystemSpec(
        preset="swarm",
        fill={
            "personas": {
                "hawk": AgentSpec(prompt="You care about missile defence budgets."),
                "dove": AgentSpec(prompt="You care about orphan poetry festivals."),
            }
        },
    )
    board = _board(spec, preset)
    cast = ["hawk", "dove"]

    class Themed:
        """Answers in one persona's vocabulary, so the ranking has something to prefer."""

        def __init__(self) -> None:
            self.seen: dict[str, list[str]] = {}

        async def complete(self, call: Call, view: View) -> str:
            who = "hawk" if "missile" in call.prompt else "dove"
            self.seen.setdefault(who, []).append(call.input)
            return (
                "missile defence budgets rise"
                if who == "hawk"
                else "poetry festivals bloom"
            )

    llm = Themed()
    await board.run(
        preset.seed(TOPIC, cast), goal="__never__", bind={"llm": llm}, budget=3
    )

    assert "missile" in llm.seen["hawk"][-1]
    assert "poetry" in llm.seen["dove"][-1]
