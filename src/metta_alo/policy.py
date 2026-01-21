from typing import Optional


def parse_policy_identifier(identifier: str) -> tuple[str, Optional[int]]:
    """Parse a policy identifier into (name, version)."""
    if identifier.endswith(":latest"):
        return identifier[:-7], None
    if ":" not in identifier:
        return identifier, None
    name, version_str = identifier.rsplit(":", 1)
    if not name:
        raise ValueError(f"Invalid policy identifier: {identifier}")
    version_str = version_str.lstrip("v")
    if not version_str.isdigit():
        raise ValueError(f"Invalid version format: {identifier}")
    return name, int(version_str)
