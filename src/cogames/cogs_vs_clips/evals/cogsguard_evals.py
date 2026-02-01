from __future__ import annotations

from pathlib import Path

from cogames.cogs_vs_clips.mission import CogsGuardMission, Site
from cogames.cogs_vs_clips.mission_utils import get_map

MAPS_DIR = Path(__file__).resolve().parent.parent.parent / "maps"

COGSGUARD_EVALS_BASE = Site(
    name="cogsguard_evals",
    description="CogsGuard evaluation arenas.",
    map_builder=get_map("evals/eval_balanced_spread.map"),
    min_cogs=1,
    max_cogs=20,
)


def _count_spawn_pads(map_path: Path) -> int:
    text = map_path.read_text()
    if "map_data:" not in text:
        raise ValueError(f"Missing map_data block in {map_path}")
    map_section = text.split("map_data:", 1)[1].split("char_to_map_name:", 1)[0]
    count = map_section.count("@")
    if count <= 0:
        raise ValueError(f"No spawn pads found in {map_path}")
    return count


def _make_eval_site(map_name: str, num_cogs: int) -> Site:
    site = COGSGUARD_EVALS_BASE.model_copy(
        update={
            "map_builder": get_map(map_name),
            "min_cogs": num_cogs,
            "max_cogs": num_cogs,
        }
    )
    return site


def _description_from_stem(stem: str) -> str:
    display = stem
    if display.startswith("eval_"):
        display = display[len("eval_") :]
    display = display.replace("_", " ")
    return f"CogsGuard eval: {display}."


COGSGUARD_EVAL_MAPS: list[str] = [
    "evals/eval_balanced_spread.map",
    "evals/eval_clip_oxygen.map",
    "evals/eval_collect_resources.map",
    "evals/eval_collect_resources_medium.map",
    "evals/eval_collect_resources_hard.map",
    "evals/eval_divide_and_conquer.map",
    "evals/eval_energy_starved.map",
    "evals/eval_multi_coordinated_collect_hard.map",
    "evals/eval_oxygen_bottleneck.map",
    "evals/eval_single_use_world.map",
    "evals/extractor_hub_30x30.map",
    "evals/extractor_hub_50x50.map",
    "evals/extractor_hub_70x70.map",
    "evals/extractor_hub_80x80.map",
    "evals/extractor_hub_100x100.map",
]

COGSGUARD_EVAL_COGS = {map_name: _count_spawn_pads(MAPS_DIR / map_name) for map_name in COGSGUARD_EVAL_MAPS}

COGSGUARD_EVAL_MISSIONS: list[CogsGuardMission] = []
for map_name in COGSGUARD_EVAL_MAPS:
    stem = Path(map_name).stem
    num_cogs = COGSGUARD_EVAL_COGS[map_name]
    site = _make_eval_site(map_name, num_cogs)
    COGSGUARD_EVAL_MISSIONS.append(
        CogsGuardMission(
            name=stem,
            description=_description_from_stem(stem),
            site=site,
            num_cogs=num_cogs,
        )
    )
