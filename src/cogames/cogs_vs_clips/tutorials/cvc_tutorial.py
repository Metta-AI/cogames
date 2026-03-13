"""CvC tutorial mission configuration."""

from cogames.games.cogs_vs_clips.game.clips import ClipsVariant
from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.terrain import MachinaArena
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.compound import CompoundConfig

CVC_TUTORIAL_MAP_BUILDER = MapGen.Config(
    width=35,
    height=35,
    instance=MachinaArena.Config(
        spawn_count=1,
        building_coverage=0.05,
        hub=CompoundConfig(
            hub_object="c:hub",
            corner_bundle="extractors",
            cross_bundle="none",
            cross_distance=5,
            hub_width=15,
            hub_height=15,
            outer_clearance=2,
            stations=[
                "c:aligner",
                "c:scrambler",
                "c:miner",
                "c:scout",
            ],
        ),
    ),
)

CvCTutorialMission = CvCMission(
    name="cvc_tutorial",
    description="Learn the basics of CvC: Roles, Resources, and Territory Control.",
    map_builder=CVC_TUTORIAL_MAP_BUILDER,
    num_cogs=1,
    min_cogs=1,
    max_cogs=1,
    max_steps=2000,
).with_variants([ClipsVariant()])
