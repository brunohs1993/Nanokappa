# Copyright (C) 2023-2024, Bruno Hartmann da Silva
# License: MIT

from pydantic import BaseModel as PdtBaseModel, ConfigDict


class BaseModel(PdtBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
