from __future__ import annotations


ALIGNER_SKILL_DESCRIPTIONS = {
    "gear_up": "Route to the aligner station and acquire aligner gear.",
    "get_heart": "Route to the hub and obtain a heart if possible.",
    "align_neutral": "Route to a known alignable neutral junction and align it.",
    "explore": "Wander to reveal more map area and discover new neutral junctions or routes.",
    "unstuck": "Try a short escape pattern to recover from repeated blocked moves, then hand control back for replanning.",
}


def build_llm_aligner_prompt(
    *,
    has_aligner: bool,
    has_heart: bool,
    hub_visible: bool,
    known_neutral_junctions: int,
    known_friendly_junctions: int,
    current_skill: str | None,
    no_move_steps: int,
    recent_events: list[str],
) -> str:
    skills = "\n".join(f"- {name}: {description}" for name, description in ALIGNER_SKILL_DESCRIPTIONS.items())
    events = "\n".join(f"- {event}" for event in recent_events[-6:]) or "- none"
    return (
        "You control one aligner cog in CoGames. Maximize aligned_junction_held by capturing neutral junctions early and repeatedly.\n"
        "Choose exactly one next skill from the available skills.\n"
        "Valid skill names are exactly: gear_up, get_heart, align_neutral, explore, unstuck. Do not invent new names.\n"
        "Preconditions:\n"
        "- If has_aligner is false, prefer gear_up.\n"
        "- Do not choose align_neutral unless has_aligner is true and has_heart is true.\n"
        "- If has_aligner is true and has_heart is false, prefer get_heart unless exploring is clearly necessary to discover a target.\n"
        "Respond as JSON like {\"skill\": \"align_neutral\", \"reason\": \"...\"}.\n\n"
        f"Available skills:\n{skills}\n\n"
        f"State:\n"
        f"- has_aligner: {has_aligner}\n"
        f"- has_heart: {has_heart}\n"
        f"- hub_visible: {hub_visible}\n"
        f"- known_neutral_junctions: {known_neutral_junctions}\n"
        f"- known_friendly_junctions: {known_friendly_junctions}\n"
        f"- current_skill: {current_skill or 'none'}\n"
        f"- no_move_steps: {no_move_steps}\n\n"
        f"Recent events:\n{events}\n"
    )
