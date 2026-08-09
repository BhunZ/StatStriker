"""Team-name resolution.

Two sources name the same club differently — FBRef writes "Manchester Utd", Understat
writes "Manchester United". Every merge in this pipeline joins on the resolved name, so a
resolver that silently returns the wrong club corrupts a team's entire history and nothing
downstream would notice: the row count stays right, the goals stay plausible, and one
club's form quietly belongs to another.
"""

import pytest

from processors.entity_resolver import ALIAS_MAP, CANONICAL_TEAMS, EntityResolver


@pytest.fixture
def resolver():
    return EntityResolver()


def test_a_canonical_name_resolves_to_itself(resolver):
    for team in CANONICAL_TEAMS[:5]:
        assert resolver.resolve(team) == team


def test_resolution_is_case_insensitive(resolver):
    assert resolver.resolve("arsenal") == resolver.resolve("ARSENAL") == "Arsenal"


@pytest.mark.parametrize("written, expected", [
    ("Manchester Utd", "Manchester United"),
    ("Man United", "Manchester United"),
    ("Man City", "Manchester City"),
    ("Spurs", "Tottenham"),
    ("Newcastle Utd", "Newcastle United"),
    ("Nott'ham Forest", "Nottingham Forest"),
])
def test_the_spellings_the_sources_actually_use(resolver, written, expected):
    assert resolver.resolve(written) == expected


def test_every_alias_points_at_a_canonical_team():
    """An alias mapping to a name not in CANONICAL_TEAMS creates a club that exists in the
    merge and nowhere else."""
    unknown = {v for v in ALIAS_MAP.values() if v not in CANONICAL_TEAMS}
    assert unknown == set(), f"aliases resolve to unknown clubs: {sorted(unknown)}"


def test_no_two_canonical_teams_are_the_same_club():
    assert len(CANONICAL_TEAMS) == len(set(CANONICAL_TEAMS))


def test_a_typo_still_finds_the_right_club(resolver):
    """Fuzzy matching is the third tier; it exists for spellings nobody curated."""
    assert resolver.resolve("Arsenl") == "Arsenal"
    assert resolver.resolve("Liverpol") == "Liverpool"


def test_something_that_is_not_a_club_is_refused_rather_than_guessed(resolver):
    """Returning a plausible-looking club for junk input is the dangerous failure: it
    merges cleanly and is invisible afterwards."""
    with pytest.raises(ValueError):
        resolver.resolve("Definitely Not A Football Club 12345")


def test_two_different_clubs_never_collapse_into_one(resolver):
    """Manchester City and Manchester United are the only pair in CANONICAL_TEAMS sharing a
    word, which makes them the one place fuzzy matching could merge two real clubs."""
    for city in ("Manchester City", "Man City", "Man. City"):
        for united in ("Manchester United", "Manchester Utd", "Man United", "Man Utd"):
            assert resolver.resolve(city) != resolver.resolve(united), \
                f"{city!r} and {united!r} resolved to the same club"


def test_every_canonical_team_survives_a_round_trip(resolver):
    """Whatever else the resolver does, it must be idempotent on its own output."""
    for team in CANONICAL_TEAMS:
        assert resolver.resolve(resolver.resolve(team)) == team
