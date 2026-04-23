from __future__ import annotations


def _clean(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def recruiting_player_query(player_name: str) -> str:
    player = _clean(player_name) or "player"
    return f"{player} football recruiting high school scouting report official visit"


def recruiting_team_query(team_name: str) -> str:
    team = _clean(team_name) or "team"
    return f"{team} football program roster competition coaching recruiting"


def transfer_player_query(player_name: str) -> str:
    player = _clean(player_name) or "player"
    return (
        f"{player} football transfer portal intent eligibility injury scouting report draft"
    )


def transfer_team_query(team_name: str) -> str:
    team = _clean(team_name) or "team"
    return (
        f"{team} football program roster competition coaching transfer portal"
    )