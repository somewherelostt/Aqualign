"""
Dot Product Tesseract - Vector similarity computation

This tesseract takes two vectors and computes:
- Dot product between them
- Cosine similarity (normalized dot product)
"""

from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType
from typing_extensions import Self

import numpy as np


class InputSchema(BaseModel):
    """Input schema for the dot product tesseract"""
    vector_a: Differentiable[Array[(None,), Float32]] = Field("An arbitrary vector")
    vector_b: Differentiable[Array[(None,), Float32]] = Field("An arbitrary vector")

    @model_validator(mode="after")
    def validate_shape_inputs(self) -> Self:
        if self.vector_a.shape != self.vector_b.shape:
            raise ValueError(
                f"Inputs vectors must have the same shape. "
                f"Got {self.vector_a.shape} and {self.vector_b.shape} instead."
            )
        return self


class OutputSchema(BaseModel):
    """Output schema for the dot product tesseract"""
    dot_product: Differentiable[Float32]
    cosine_similarity: Differentiable[Float32]


def apply(inputs: InputSchema) -> OutputSchema:
    vector_a = inputs.vector_a
    vector_b = inputs.vector_b

    # compute dot product
    dot_prod = np.dot(vector_a, vector_b)

    # Compute magnitudes for cosine similarity
    # Add small epsilon for numerical stability
    eps = 1e-8
    magnitude_a = np.sqrt(np.sum(vector_a ** 2) + eps)
    magnitude_b = np.sqrt(np.sum(vector_b ** 2) + eps)

    # Compute cosine similarity
    cosine_sim = dot_prod / (magnitude_a * magnitude_b)

    # Create and return output
    output = OutputSchema(
        dot_product=float(dot_prod),
        cosine_similarity=float(cosine_sim)
    )

    return output

def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""

    return {
        "dot_product": ShapeDType(shape=(), dtype="float32"),
        "cosine_similarity": ShapeDType(shape=(), dtype="float32")
    }
