# -*- coding: utf-8 -*-
"""Regression tests for neural vocal segmentation."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vocalpy.errors import ValidationError
from vocalpy.nn.segmenter import VocalSegmenter


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_device = None
        self.eval_called = False

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def forward(self, image):
        return image


def test_segmenter_loads_in_memory_model_to_requested_device():
    segmenter = VocalSegmenter.__new__(VocalSegmenter)
    model = FakeModel()

    returned_model = segmenter.load_segmentation_model("cuda", model=model)

    assert returned_model is model
    assert model.to_device == "cuda"
    assert model.eval_called is True


def test_segmenter_rejects_missing_model_source():
    segmenter = VocalSegmenter.__new__(VocalSegmenter)

    with pytest.raises(ValidationError, match="segmentation model path is required"):
        segmenter.load_segmentation_model("cpu")


def test_segment_list_of_vocals_thresholds_and_resizes_predictions():
    segmenter = VocalSegmenter.__new__(VocalSegmenter)
    segmenter.threshold = 0.5
    segmenter.output_shape = (4, 4)
    segmenter.device = "cpu"
    segmenter.dataloader = [torch.zeros((1, 3, 224, 224), dtype=torch.float32)]
    segmenter.model = lambda image: torch.tensor([[[[0.0, 1.0], [2.0, -2.0]]]], dtype=torch.float32)

    result = segmenter.segment_list_of_vocals(SimpleNamespace(number_of_vocals=1))

    assert result.shape == (1, 4, 4)
    assert result.dtype == np.uint8
    assert np.all(result[0, :2, :2] == 255)
    assert np.all(result[0, :2, 2:] == 255)
    assert np.all(result[0, 2:, :2] == 255)
    assert np.all(result[0, 2:, 2:] == 0)


def test_segment_list_of_vocals_returns_empty_predictions_for_empty_input():
    segmenter = VocalSegmenter.__new__(VocalSegmenter)
    segmenter.output_shape = (3, 5)

    result = segmenter.segment_list_of_vocals(SimpleNamespace(number_of_vocals=0))

    assert result.shape == (0, 3, 5)
