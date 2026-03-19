from __future__ import annotations


SKILL_DESCRIPTIONS = {
    "mine_until_full": "Acquire miner gear if needed, then mine nearby extractors until cargo is full or progress stalls.",
    "deposit_to_hub": "Return carried resources to the hub using visible hub cues and remembered hub position.",
    "unstuck": "Try a short escape pattern to recover from repeated blocked moves, then hand control back for replanning.",
}


def build_llm_miner_prompt(
    *,
    carried_total: int,
    return_load: int,
    has_miner: bool,
    hub_visible: bool,
    remembered_hub: tuple[int | None, int | None],
    current_skill: str | None,
    no_move_steps: int,
    recent_events: list[str],
) -> str:
    skills = "\n".join(f"- {name}: {description}" for name, description in SKILL_DESCRIPTIONS.items())
    events = "\n".join(f"- {event}" for event in recent_events[-6:]) or "- none"
    hub_row, hub_col = remembered_hub
    remembered_hub_text = (
        "unknown" if hub_row is None or hub_col is None else f"spawn_relative_row={hub_row}, spawn_relative_col={hub_col}"
    )
    return (
        "You control one miner cog in CoGames. Maximize deposited resources.\n"
        "Choose exactly one next skill from the available skills.\n"
        "Respond as JSON like {\"skill\": \"mine_until_full\", \"reason\": \"...\"}.\n\n"
        f"Available skills:\n{skills}\n\n"
        f"State:\n"
        f"- has_miner: {has_miner}\n"
        f"- carried_total: {carried_total}\n"
        f"- return_load: {return_load}\n"
        f"- hub_visible: {hub_visible}\n"
        f"- remembered_hub: {remembered_hub_text}\n"
        f"- current_skill: {current_skill or 'none'}\n"
        f"- no_move_steps: {no_move_steps}\n\n"
        f"Recent events:\n{events}\n"
    )
