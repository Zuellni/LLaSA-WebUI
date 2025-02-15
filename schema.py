import random
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticUndefined


class Query(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    input: Annotated[str, Field(alias="text", min_length=1)]
    voice: Annotated[str, Field(alias="audio", default="", to_lower=True)]

    format: Annotated[
        Literal["flac", "mp3", "ogg", "wav"],
        Field(alias="response_format", default="wav"),
    ]

    max_len: Annotated[int, Field(default=300, ge=1)]
    rate: Annotated[int, Field(default=16000)]

    join: Annotated[bool, Field(default=False)]
    reuse: Annotated[bool, Field(default=False)]
    seed: Annotated[int, Field(default=-1)]

    penalty: Annotated[float, Field(default=1.0, ge=0.0)]
    temperature: Annotated[float, Field(default=1.0, ge=0.0)]
    top_k: Annotated[int, Field(default=50, ge=0)]
    top_p: Annotated[float, Field(default=1.0, ge=0.0)]

    @field_validator("join", "reuse", mode="before")
    def validate_bool(cls, value: bool | str) -> bool:
        if isinstance(value, str):
            return value.lower() == "true"

        return bool(value)

    @field_validator("seed", mode="after")
    def validate_seed(cls, value: int) -> int:
        return random.randint(0, 2**64 - 1) if value < 0 else value

    @staticmethod
    def defaults() -> dict[str, Any]:
        return {
            k: v.default
            for k, v in __class__.model_fields.items()
            if v.default is not PydanticUndefined
        }

    @staticmethod
    def formats() -> list[str]:
        return sorted(get_args(__class__.model_fields["format"].annotation))
