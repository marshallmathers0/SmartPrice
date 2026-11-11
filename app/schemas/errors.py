from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Body of every error returned by the global exception handlers.

    One shape for every error means clients handle failures in one place:
    show ``detail`` to humans, branch on ``code`` in code.
    """
    detail: str = Field(
        description="Human-readable error message",
        examples=["Product 'SKU-99999' was not found"],
    )
    code: str = Field(
        description="Stable machine-readable error code",
        examples=["product_not_found"],
    )
