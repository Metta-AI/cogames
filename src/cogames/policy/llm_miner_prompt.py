from __future__ import annotations


SKILL_DESCRIPTIONS = {
    "gear_up": "Route to the miner station and acquire miner gear.",
    "mine_until_full": "Acquire miner gear if needed, then route to known extractors and mine until cargo is full.",
    "deposit_to_hub": "Route to the hub using map memory and deposit carried resources.",
    "explore": "Move to a frontier of the known map to reveal new territory and discover new extractors or routes.",
    "unstuck": "Try a short escape pattern to recover from repeated blocked moves, then hand control back for replanning.",
}


def build_llm_miner_prompt(
    *,
    carried_total: int,
    return_load: int,
    has_miner: bool,
    hub_visible: bool,
    remembered_hub: tuple[int | None, int | None],
    known_extractors: int,
    frontier_count: int,
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
        "Valid skill names are exactly: gear_up, mine_until_full, deposit_to_hub, explore, unstuck. Do not invent new names.\n"
        "Preconditions:\n"
        "- If has_miner is false, prefer gear_up.\n"
        "- Do not choose mine_until_full or deposit_to_hub unless has_miner is true.\n"
        "- If carried_total >= return_load, prefer deposit_to_hub.\n"
        "Respond as JSON like {\"skill\": \"mine_until_full\", \"reason\": \"...\"}.\n\n"
        f"Available skills:\n{skills}\n\n"
        f"State:\n"
        f"- has_miner: {has_miner}\n"
        f"- carried_total: {carried_total}\n"
        f"- return_load: {return_load}\n"
        f"- hub_visible: {hub_visible}\n"
        f"- remembered_hub: {remembered_hub_text}\n"
        f"- known_extractors: {known_extractors}\n"
        f"- frontier_count: {frontier_count}\n"
        f"- current_skill: {current_skill or 'none'}\n"
        f"- no_move_steps: {no_move_steps}\n\n"
        f"Recent events:\n{events}\n"
    )
