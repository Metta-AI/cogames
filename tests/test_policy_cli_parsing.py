from pathlib import Path

import pytest

from cogames.cli.policy import parse_policy_spec
from mettagrid.policy.loader import resolve_policy_class_path


def test_parse_policy_spec_with_class_only():
    spec = parse_policy_spec("class=random")

    assert spec.class_path == resolve_policy_class_path("random")
    assert spec.data_path is None
    assert spec.proportion == 1.0
    assert spec.init_kwargs == {}


def test_parse_policy_spec_with_shorthand_class():
    spec = parse_policy_spec("random")

    assert spec.class_path == resolve_policy_class_path("random")
    assert spec.data_path is None
    assert spec.proportion == 1.0
    assert spec.init_kwargs == {}


def test_parse_policy_spec_with_data_proportion_and_kwargs(tmp_path: Path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_text("dummy")

    spec = parse_policy_spec(f"random,data={checkpoint},proportion=0.5,kw.alpha=0.1,kw.beta=value,kw.with-hyphen=ok")

    assert spec.class_path == resolve_policy_class_path("random")
    assert spec.data_path == str(checkpoint.resolve())
    assert spec.proportion == 0.5
    assert spec.init_kwargs == {"alpha": "0.1", "beta": "value", "with_hyphen": "ok"}


def test_parse_policy_spec_with_metta_uri_query_params():
    """metta:// URIs with query params should parse correctly."""
    spec = parse_policy_spec("metta://policy/random?vibe_action_p=0.01")

    assert spec.class_path == resolve_policy_class_path("random")
    assert spec.init_kwargs == {"vibe_action_p": "0.01"}


def test_parse_policy_spec_with_metta_uri_query_commas():
    """metta:// URIs should allow commas inside query values."""
    spec = parse_policy_spec("metta://policy/random?role_vibes=miner,scout")

    assert spec.class_path == resolve_policy_class_path("random")
    assert spec.init_kwargs == {"role_vibes": "miner,scout"}


def test_parse_policy_spec_with_metta_uri_and_proportion():
    """metta:// URIs can be combined with proportion."""
    spec = parse_policy_spec("metta://policy/random?vibe_action_p=0.5,proportion=0.25")

    assert spec.class_path == resolve_policy_class_path("random")
    assert spec.init_kwargs == {"vibe_action_p": "0.5"}
    assert spec.proportion == 0.25


@pytest.mark.parametrize(
    "raw_spec",
    [
        "",
        "data=only",
        "random:train_dir/model.pt",
        "random,proportion=-1",
    ],
)
def test_parse_policy_spec_rejects_invalid_input(raw_spec: str):
    with pytest.raises(ValueError):
        parse_policy_spec(raw_spec)
