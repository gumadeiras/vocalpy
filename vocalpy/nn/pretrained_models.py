# -*- coding: utf-8 -*-
"""Metadata and validation helpers for bundled pretrained models."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import hashlib

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PretrainedModelSpec:
    network_type: str
    filename: str
    num_classes: int
    classes: tuple[str, ...]
    sha256: str

    @property
    def path(self) -> Path:
        return Path(__file__).with_name("pretrained") / self.filename


PRETRAINED_MODEL_SPECS = {
    "noise": PretrainedModelSpec(
        network_type="noise",
        filename="noise_model.pth.tar",
        num_classes=2,
        classes=("noise", "vocal"),
        sha256="454ed81137edfe22c2908185499ff67e7217be45d60d0ac4f6264e3d256e106a",
    ),
    "class": PretrainedModelSpec(
        network_type="class",
        filename="class_model.pth.tar",
        num_classes=11,
        classes=(
            "chevron",
            "complex",
            "down_fm",
            "flat",
            "mult_steps",
            "rev_chevron",
            "short",
            "step_down",
            "step_up",
            "two_steps",
            "up_fm",
        ),
        sha256="223cb63c5295284d67298a9cdddaa24e57df07e7c826d3cd6c23ed66a503e939",
    ),
}


def get_pretrained_model_spec(network_type: str) -> PretrainedModelSpec:
    try:
        return PRETRAINED_MODEL_SPECS[network_type]
    except KeyError as exc:
        raise ValueError(f"unsupported network_type: {network_type}") from exc


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pretrained_model_file(path: str | Path, expected_sha256: str | None = None) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pretrained model does not exist: {path}")

    actual_sha256 = compute_sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"pretrained model checksum mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256
