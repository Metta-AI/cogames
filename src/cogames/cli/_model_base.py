from pydantic import BaseModel, ConfigDict


class CLIModel(BaseModel):
    """Base class for all CLI response models. Uses extra="allow" for forward compatibility."""

    model_config = ConfigDict(extra="allow")
