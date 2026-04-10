# -*- coding: utf-8 -*-
"""Regression tests for recording and visualization helpers."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from types import SimpleNamespace

import numpy as np
import pytest

from vocalpy.errors import ConfigurationError, RecordingStateError, ValidationError
from vocalpy.modules.recording import Recording
from vocalpy.modules.viz import SingleViz, Viz


def test_recording_create_animal_pipeline_raises_when_identifier_disabled():
    recording = Recording.__new__(Recording)
    recording.params = {"identifier": False}
    recording._animal_name = "mouse"

    with pytest.raises(ConfigurationError, match="animal pipeline is not available"):
        recording.create_animal_pipeline()


def test_recording_save_recording_data_to_csv_respects_explicit_path(monkeypatch, tmp_path):
    recording = Recording.__new__(Recording)
    recording._list_of_vocals = None
    recording.recording_name = "mouse.wav"
    recording.output_dir = str(tmp_path / "default")
    list_of_vocals = SimpleNamespace(intervals_fixed=True)
    captured = {}

    monkeypatch.setattr(
        "vocalpy.modules.recording.create_dataframe_from_list_of_vocals",
        lambda received_list: {"received": received_list},
    )

    def fake_save_dataframe_as_csv(dataframe, path, filename):
        captured["dataframe"] = dataframe
        captured["path"] = path
        captured["filename"] = filename

    monkeypatch.setattr("vocalpy.modules.recording.save_dataframe_as_csv", fake_save_dataframe_as_csv)

    recording.save_recording_data_to_csv(list_of_vocals=list_of_vocals, path=str(tmp_path / "target"))

    assert captured == {
        "dataframe": {"received": list_of_vocals},
        "path": str(tmp_path / "target"),
        "filename": "mouse.wav",
    }


def test_recording_save_spectrograms_requires_vocals():
    recording = Recording.__new__(Recording)
    recording._list_of_vocals = None

    with pytest.raises(RecordingStateError, match="recording has no list of vocals"):
        recording.save_spectrograms()


def test_recording_create_paths_only_creates_output_dir(tmp_path):
    recording = Recording.__new__(Recording)
    recording_path = tmp_path / "mouse.wav"
    recording_path.write_bytes(b"")

    recording.create_paths(str(recording_path))

    assert (tmp_path / "mouse_outputs").is_dir()
    assert not (tmp_path / "mouse_outputs" / "spectrogram").exists()
    assert not (tmp_path / "mouse_outputs" / "mask").exists()


def test_recording_update_vocals_with_class_classification_raises_on_prediction_mismatch():
    recording = Recording.__new__(Recording)
    recording._list_of_vocals = SimpleNamespace(number_of_vocals=1)

    with pytest.raises(RecordingStateError, match="number of vocals and classifier predictions differ"):
        recording.update_vocals_with_class_classification(np.empty((0, 2)), ["flat", "up_fm"])


def test_recording_update_vocals_with_segmentation_masks_raises_on_prediction_mismatch():
    recording = Recording.__new__(Recording)
    recording._list_of_vocals = SimpleNamespace(number_of_vocals=1)

    with pytest.raises(RecordingStateError, match="number of vocals and segmenter predictions differ"):
        recording.update_vocals_with_segmentation_masks(np.empty((0, 2, 2), dtype=np.uint8))


def test_recording_segment_vocalizations_uses_optional_model_path_when_enabled():
    recording = Recording.__new__(Recording)
    recording.params = {"segmenter": True, "segmentation_model_path": None}
    recording._animal = SimpleNamespace(_animal="mouse")
    recording._list_of_vocals = SimpleNamespace(number_of_vocals=1)
    captured = {}

    def fake_segment_vocalizations(list_of_vocals, path_to_model, threshold):
        captured["list_of_vocals"] = list_of_vocals
        captured["path_to_model"] = path_to_model
        captured["threshold"] = threshold
        return np.ones((1, 2, 2), dtype=np.uint8)

    recording._animal.segment_vocalizations = fake_segment_vocalizations
    recording.update_vocals_with_segmentation_masks = lambda predictions: captured.setdefault("predictions", predictions)

    recording.segment_vocalizations()

    assert captured["path_to_model"] is None
    assert captured["threshold"] is None
    assert captured["list_of_vocals"] is recording._list_of_vocals


def test_viz_initialization_derives_group_names_from_recording_paths(monkeypatch):
    monkeypatch.setattr(Viz, "create_viz_for_each_recording", lambda self: 0)
    monkeypatch.setattr(Viz, "create_group_viz", lambda self: 0)

    viz = Viz([["/tmp/mouse_1_outputs/recording_without_spectrograms.vocalpy"]])

    assert viz._group_names.tolist() == ["mouse_1"]
    assert viz._number_of_groups == 1


def test_viz_combine_viz_dataframes_requires_non_empty_list():
    viz = Viz.__new__(Viz)

    with pytest.raises(ValidationError, match="non-empty list_of_viz"):
        viz.combine_viz_dataframes()


def test_single_viz_requires_recording_path():
    with pytest.raises(ValidationError, match="recording_path"):
        SingleViz()
