"""Layout variants for Overcogged."""

from __future__ import annotations

from cogames.core import CoGameMissionVariant

LINE_STATIONS = [
    "veg_station",
    "meat_station",
    "plate_station",
    "chopping_station",
    "cooking_station",
    "fryer_station",
    "serving_station",
    "wash_station",
    "order_board",
]

REVERSE_STATIONS = list(reversed(LINE_STATIONS))

SERVICE_SIDE_STATIONS = [
    "order_board",
    "serving_station",
    "plate_station",
    "wash_station",
    "chopping_station",
    "veg_station",
    "meat_station",
    "cooking_station",
    "fryer_station",
]


class TightHubVariant(CoGameMissionVariant):
    name: str = "tight_hub"
    description: str = "Use tighter hub layout with deterministic spawns."

    def modify_mission(self, mission) -> None:
        mission.hub_layout = "tight"
        mission.randomize_spawn_positions = False


class LayoutLineVariant(CoGameMissionVariant):
    name: str = "layout_line"
    description: str = "Wide default line layout for onboarding."

    def modify_mission(self, mission) -> None:
        mission.hub_layout = "default"
        mission.hub_width = 23
        mission.hub_height = 23
        mission.station_order = list(LINE_STATIONS)
        mission.randomize_spawn_positions = False


class LayoutReverseVariant(CoGameMissionVariant):
    name: str = "layout_reverse"
    description: str = "Reversed station line to force traversal adaptation."

    def modify_mission(self, mission) -> None:
        mission.hub_layout = "default"
        mission.hub_width = 23
        mission.hub_height = 23
        mission.station_order = list(REVERSE_STATIONS)
        mission.randomize_spawn_positions = False


class LayoutCompactVariant(CoGameMissionVariant):
    name: str = "layout_compact"
    description: str = "Compact default layout with service-side station cluster."

    def modify_mission(self, mission) -> None:
        mission.hub_layout = "default"
        mission.hub_width = 19
        mission.hub_height = 19
        mission.station_order = list(SERVICE_SIDE_STATIONS)
        mission.randomize_spawn_positions = False


class LayoutTightFlowVariant(CoGameMissionVariant):
    name: str = "layout_tight_flow"
    description: str = "Tight layout with deterministic spawns and line flow."

    def modify_mission(self, mission) -> None:
        mission.hub_layout = "tight"
        mission.hub_width = 19
        mission.hub_height = 19
        mission.station_order = list(LINE_STATIONS)
        mission.randomize_spawn_positions = False
