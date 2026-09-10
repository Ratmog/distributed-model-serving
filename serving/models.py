from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)


class Generate(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=32)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0, ge=0, le=2)

    @model_validator(mode="after")
    def bound_total(self):
        if sum(len(m.content) for m in self.messages) > 32000:
            raise ValueError("Total message content exceeds 32000 characters")
        return self
