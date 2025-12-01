"""
Main Pipeline - Tesseract Hackathon Template

Demonstrates two tesseracts working together:
1. scaler - Scales input vectors
2. dotproduct - Computes similarity between vectors
"""

import jax.numpy as jnp
from jax import Array
from tesseract_core import Tesseract


def path1_separate_calls(scaler: Tesseract, dotproduct: Tesseract, vec_a: Array, vec_b: Array) -> None:
    """Path 1: Call tesseracts separately with intermediate results"""
    print("\n" + "="*60)
    print("PATH 1: Calling tesseracts separately")
    print("="*60)

    print(f"\nInput vectors: {vec_a}, {vec_b}")

    # Scale vector A
    output_a = scaler.apply({
        "vector": vec_a, "scale_factor": jnp.array(2.0)
    })
    scaled_a = output_a['scaled']
    print(f"Scaled A: {scaled_a}")

    # Scale vector B
    output_b = scaler.apply({
        "vector": vec_b, "scale_factor": jnp.array(2.0)
    })
    scaled_b = output_b['scaled']
    print(f"Scaled B: {scaled_b}")

    # Compute similarity
    result = dotproduct.apply({
        "vector_a": scaled_a, "vector_b": scaled_b
    })
    print(f"Cosine similarity: {result['cosine_similarity']:.4f}")


def path2_composed_calls(scaler: Tesseract, dotproduct: Tesseract, vec_a: Array, vec_b: Array) -> None:
    """Path 2: Call tesseracts together in a pipeline with Tesseract-JAX"""
    from tesseract_jax import apply_tesseract

    print("\n" + "="*60)
    print("PATH 2: Calling tesseracts together")
    print("="*60)

    print(f"\nInput vectors: {vec_a}, {vec_b}")

    # Scale both and compute similarity
    out_a = apply_tesseract(scaler, {
        "vector": vec_a, "scale_factor": jnp.array(2.0)
    })
    out_b = apply_tesseract(scaler, {
        "vector": vec_b, "scale_factor": jnp.array(2.0)
    })
    result = apply_tesseract(dotproduct, {
        "vector_a": out_a['scaled'],
        "vector_b": out_b['scaled']
    })
    print(f"Cosine similarity: {result['cosine_similarity']:.4f}")


def main() -> None:
    print("\n" + "="*60)
    print("  TESSERACT DEMO: Two Interacting Tesseracts")
    print("="*60)

    # Behind the scenes tesseracts are loaded with docker compose
    # and serve an HTTP enpoint that can be called
    # at the end of the context the containers are stopped
    # it is possible to do that manually with
    #     t = Tesseract.from_image(image_name)
    #     t.serve()
    #     t.teardown()
    # However we recommend using context manager for cleaner and safer approach
    with Tesseract.from_image("scaler") as scaler, Tesseract.from_image("dotproduct_jax") as dotproduct:

        # Hardcoded test vectors
        vec_a = jnp.array([3.0, 4.0, 0.0])
        vec_b = jnp.array([1.0, 0.0, 0.0])

        # Run both paths
        path1_separate_calls(scaler, dotproduct, vec_a, vec_b)
        path2_composed_calls(scaler, dotproduct, vec_a, vec_b)

        print("\n" + "="*60)
        print("✓ Demo complete!")
        print("="*60)


if __name__ == "__main__":
    main()
