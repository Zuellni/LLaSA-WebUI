import random
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class Query(BaseModel):
    input: Annotated[str, Field(min_length=1)]
    voice: Annotated[str, Field(default="", to_lower=True)]

    format: Annotated[Literal["flac", "mp3", "ogg", "wav"], Field(default="wav")]
    max_len: Annotated[int, Field(default=200, ge=1)]
    sample_rate: Annotated[int, Field(default=16000)]

    repetition_penalty: Annotated[float, Field(default=1.0, ge=0.0)]
    temperature: Annotated[float, Field(default=1.0, ge=0.0)]
    top_k: Annotated[int, Field(default=50, ge=0)]
    top_p: Annotated[float, Field(default=1.0, ge=0.0)]

    reuse: Annotated[bool, Field(default=False)]
    seed: Annotated[int, Field(default=-1)]

    @field_validator("reuse", mode="before")
    def validate_reuse(cls, value: bool | str) -> bool:
        if isinstance(value, str):
            return value.lower() == "true"

        return bool(value)

    @field_validator("seed", mode="after")
    def validate_seed(cls, value: int) -> int:
        return random.randint(0, 2**64 - 1) if value < 0 else value
