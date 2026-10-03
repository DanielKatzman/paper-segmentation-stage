"""Deterministic training-data generation for universal paper segmentation."""

from .compiler import GeneratorConfig, generate_dataset
from .export_annotations import export_reviewed_annotations
from .sources import sync_generator_inputs

__all__ = [
    "GeneratorConfig",
    "export_reviewed_annotations",
    "generate_dataset",
    "sync_generator_inputs",
]
