from langchain.pydantic_v1 import BaseModel


class Step(BaseModel):
    """Step."""

    value: str
    """The value."""


class Plan(BaseModel):
    """Plan."""

    steps: list[Step]
    """The steps."""


class StepResponse(BaseModel):
    """Step response."""

    response: str
    """The response."""
