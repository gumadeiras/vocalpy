# -*- coding: utf-8 -*-
"""Regression tests for modern torch/torchvision classifier behavior."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from types import SimpleNamespace

import numpy as np
import pytest

from vocalpy.nn.classifier import VocalClassifier


class FakeModel:
    def __init__(self):
        self.to_device = None
        self.eval_called = False

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class FakeImage:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakePrediction:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.cpu_called = False

    def cpu(self):
        self.cpu_called = True
        return self

    def numpy(self):
        if not self.cpu_called:
            raise AssertionError("prediction must be moved to CPU before numpy()")
        return self.values


def test_load_pretrained_noise_model_moves_model_to_requested_device(monkeypatch):
    classifier = VocalClassifier.__new__(VocalClassifier)
    model = FakeModel()

    monkeypatch.setattr(VocalClassifier, "build_mobilenet_v2_classifier", staticmethod(lambda num_classes: model))
    monkeypatch.setattr("vocalpy.nn.classifier.load_checkpoint", lambda path, model_obj, device: None)

    returned_model = classifier.load_pretrained_noise_model("cuda", "custom-checkpoint.pth.tar")

    assert returned_model is model
    assert model.to_device == "cuda"
    assert model.eval_called is True
    assert classifier.classes == ["noise", "vocal"]


def test_classify_list_of_vocals_class_moves_predictions_to_cpu(monkeypatch):
    classifier = VocalClassifier.__new__(VocalClassifier)
    classifier.device = "cuda"
    classifier.dataloader = [FakeImage()]
    classifier.model = lambda image: "score"

    prediction = FakePrediction([[0.1, 0.9]])
    monkeypatch.setattr("vocalpy.nn.classifier.softmax", lambda score, dim: prediction)

    result = classifier.classify_list_of_vocals_class(SimpleNamespace(number_of_vocals=1))

    assert prediction.cpu_called is True
    assert np.allclose(result, [[0.1, 0.9]])


def test_classify_list_of_vocals_noise_moves_predictions_to_cpu(monkeypatch):
    classifier = VocalClassifier.__new__(VocalClassifier)
    classifier.device = "cuda"
    classifier.dataloader = [FakeImage()]
    classifier.model = lambda image: "score"

    prediction = FakePrediction([1])
    monkeypatch.setattr("vocalpy.nn.classifier.torch.max", lambda score, dim: (None, prediction))

    result = classifier.classify_list_of_vocals_noise(SimpleNamespace(number_of_vocals=1))

    assert prediction.cpu_called is True
    assert result.tolist() == [True]
