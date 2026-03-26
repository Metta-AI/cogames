"""Tests for assay CLI helpers."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import typer

from cogames.cli.assay import _resolve_policy_version_id

_FOUND = object()


def _make_client(*, lookup_return=None):
    client = MagicMock()
    pv = MagicMock()
    pv.id = uuid4()
    if lookup_return is _FOUND:
        client.lookup_policy_version.return_value = pv
        client._pv = pv
    else:
        client.lookup_policy_version.return_value = lookup_return
    return client


def test_resolve_bare_uuid():
    client = MagicMock()
    uid = uuid4()
    assert _resolve_policy_version_id(client, str(uid)) == uid
    client.lookup_policy_version.assert_not_called()


def test_resolve_name_only():
    client = _make_client(lookup_return=_FOUND)
    result = _resolve_policy_version_id(client, "my-policy")
    client.lookup_policy_version.assert_called_once_with(name="my-policy", version=None)
    assert result == client._pv.id


def test_resolve_name_with_numeric_version():
    client = _make_client(lookup_return=_FOUND)
    _resolve_policy_version_id(client, "my-policy:3")
    client.lookup_policy_version.assert_called_once_with(name="my-policy", version=3)


def test_resolve_name_with_v_prefixed_version():
    client = _make_client(lookup_return=_FOUND)
    _resolve_policy_version_id(client, "champion-policy:v1")
    client.lookup_policy_version.assert_called_once_with(name="champion-policy", version=1)


def test_resolve_name_with_v_prefixed_multidigit():
    client = _make_client(lookup_return=_FOUND)
    _resolve_policy_version_id(client, "champion-policy:v12")
    client.lookup_policy_version.assert_called_once_with(name="champion-policy", version=12)


def test_resolve_not_found_exits():
    client = _make_client(lookup_return=None)
    with pytest.raises(typer.Exit):
        _resolve_policy_version_id(client, "nonexistent-policy")
