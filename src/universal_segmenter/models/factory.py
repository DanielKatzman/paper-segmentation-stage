"""Create the one approved trainable architecture for this stage."""

from __future__ import annotations


MODEL_TYPE = "segformer_b1_detail"


def create_model(model_type: str, pretrained: bool = False):
    if model_type != MODEL_TYPE:
        raise ValueError(
            f"unsupported model type: {model_type!r}; this release contains only "
            f"{MODEL_TYPE!r}")
    from .segformer_detail import create_segformer_detail
    return create_segformer_detail(variant="b1", pretrained=pretrained)

