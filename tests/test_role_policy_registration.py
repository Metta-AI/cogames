from __future__ import annotations

import pytest

from mettagrid.policy.loader import resolve_policy_class_path


@pytest.mark.parametrize(
    ("short_name", "class_suffix"),
    [
        ("miner", "cogames.policy.role_policies.MinerRolePolicy"),
        ("scout", "cogames.policy.role_policies.ScoutRolePolicy"),
        ("aligner", "cogames.policy.role_policies.AlignerRolePolicy"),
        ("scrambler", "cogames.policy.role_policies.ScramblerRolePolicy"),
    ],
)
def test_canonical_role_short_names_resolve_in_cogames(short_name: str, class_suffix: str) -> None:
    assert resolve_policy_class_path(short_name).endswith(class_suffix)


@pytest.mark.parametrize("legacy_name", ["role_miner", "role_scout", "role_aligner", "role_scrambler"])
def test_legacy_role_short_names_are_not_registered(legacy_name: str) -> None:
    assert resolve_policy_class_path(legacy_name) == legacy_name
