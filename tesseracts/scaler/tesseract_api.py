"""
Scaler Tesseract

This tesseract takes an input vector and a scale factor, returning the scaled vector
"""

from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType
from typing_extensions import Self


class InputSchema(BaseModel):
    """Input schema for a single vector with scale factor"""
    vector: Differentiable[Array[(None,), Float32]] = Field(
        description="An arbitrary vector"
    )
    scale_factor: Differentiable[Float32] = Field(description="A scalar to scale the vector by", default=1.0)

    @model_validator(mode="after")
    def validate_shape_inputs(self) -> Self:
        if len(self.vector.shape) != 1:
            raise ValueError(
                f"Vector must be 1-dimensional. "
                f"Got {self.vector.shape} instead."
            )
        return self


class OutputSchema(BaseModel):
    """Output schema for the scaler tesseract"""
    scaled: Differentiable[Array[(None,), Float32]] = Field("Scaled vector")


def apply(inputs: InputSchema) -> OutputSchema:
    result = inputs.vector * inputs.scale_factor

    return OutputSchema(scaled=result)


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    input_vector_shapedtype = abstract_inputs.vector

    return {
        "scaled": ShapeDType(
            shape=input_vector_shapedtype.shape,
            dtype=input_vector_shapedtype.dtype
        )
    }
