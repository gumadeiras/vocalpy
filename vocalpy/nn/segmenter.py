# -*- coding: utf-8 -*-
"""Neural-network segmentation for individual vocal spectrogram crops."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vocalpy.errors import ValidationError
from vocalpy.nn import datasets
from vocalpy.utils.io import load_model


class VocalSegmenter(object):
    """
    Segment individual vocal spectrogram crops with a PyTorch model.

    The segmenter operates on the same per-vocal crop surface used by the
    classifier. The model is expected to return a binary mask logits/probability
    map for each crop.
    """

    def __init__(self, source, batch_size=32, path_to_model=None, threshold=0.5, model=None):
        self.batch_size = batch_size
        self.path_to_model = path_to_model
        self.threshold = self._validate_threshold(threshold)

        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

        self.model = self.load_segmentation_model(self.device, path_to_model=path_to_model, model=model)
        self.dataset = self.create_dataset(source)
        self.dataloader = datasets.create_dataloader(self.dataset, self.batch_size) if len(self.dataset) > 0 else []
        self.output_shape = (self.dataset.height, self.dataset.width)

    @staticmethod
    def _validate_threshold(threshold):
        threshold = float(threshold)
        if threshold <= 0 or threshold >= 1:
            raise ValidationError(f"segmentation threshold must be between 0 and 1. provided value: {threshold}")
        return threshold

    def load_segmentation_model(self, device, path_to_model=None, model=None):
        if model is None and path_to_model is None:
            raise ValidationError("segmentation model path is required when no in-memory model is provided")

        resolved_model = model if model is not None else load_model(path_to_model, device)
        if isinstance(resolved_model, dict) and "model" in resolved_model:
            resolved_model = resolved_model["model"]
        if not isinstance(resolved_model, nn.Module):
            raise ValidationError(
                "segmentation model must resolve to a torch.nn.Module. "
                f"received type: {resolved_model.__class__.__name__}"
            )

        resolved_model = resolved_model.to(device)
        resolved_model.eval()
        return resolved_model

    def create_dataset(self, source):
        if not isinstance(source, np.ndarray):
            raise ValidationError("segmentation source must be a numpy.ndarray of vocal spectrogram crops")
        return datasets.VocalDatasetFromArray(source)

    def empty_predictions(self):
        height, width = self.output_shape
        return np.empty((0, height, width), dtype=np.uint8)

    def segment_list_of_vocals(self, list_of_vocals):
        if list_of_vocals.number_of_vocals < 1:
            return self.empty_predictions()

        predictions = []
        with torch.no_grad():
            for image in self.dataloader:
                image = image.to(self.device)
                probabilities = self._prediction_to_probability_map(self.model(image))
                predictions.append(self._resize_and_threshold(probabilities))

        return np.vstack(predictions) if predictions else self.empty_predictions()

    def _prediction_to_probability_map(self, prediction):
        if prediction.ndim == 4:
            if prediction.shape[1] == 1:
                logits = prediction[:, 0]
                return self._to_probability(logits)
            if prediction.shape[1] == 2:
                return torch.softmax(prediction, dim=1)[:, 1]

        if prediction.ndim == 3:
            return self._to_probability(prediction)

        raise ValidationError(
            "segmentation model output must have shape (N,H,W), (N,1,H,W), or (N,2,H,W). "
            f"received shape: {tuple(prediction.shape)}"
        )

    @staticmethod
    def _to_probability(prediction):
        if torch.all((prediction >= 0) & (prediction <= 1)):
            return prediction
        return torch.sigmoid(prediction)

    def _resize_and_threshold(self, prediction):
        if self.output_shape == (0, 0):
            return np.empty((prediction.shape[0], 0, 0), dtype=np.uint8)

        resized = F.interpolate(
            prediction.unsqueeze(1),
            size=self.output_shape,
            mode="nearest",
        ).squeeze(1)
        return ((resized >= self.threshold).to(dtype=torch.uint8) * 255).cpu().numpy()
