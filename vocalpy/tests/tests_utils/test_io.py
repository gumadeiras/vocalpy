# -*- coding: utf-8 -*-
"""Behavior tests for I/O helpers."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from pathlib import Path

import numpy as np
import pytest
import torch

from vocalpy.errors import InputPathError
from vocalpy.utils.io import (
    create_directory,
    create_output_directory_structure,
    get_output_directory_for_audio_file,
    load_checkpoint,
    load_model,
    load_pickle_file,
    load_recording_data,
    parse_input_path,
    remove_directory,
    write_pickle_file,
)


def test_write_and_load_pickle_file_round_trip(tmp_path):
    payload = {"detected_vocals": 3}

    write_pickle_file(payload, "recording", tmp_path)
    loaded = load_pickle_file("recording", tmp_path)

    assert loaded == payload


def test_load_recording_data_round_trip(tmp_path):
    path = tmp_path / "recording.npy"
    expected = np.asarray([1, 2, 3])
    np.save(path, expected)

    loaded = load_recording_data(path)

    assert np.array_equal(loaded, expected)


def test_load_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(0.25)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optim_dict": optimizer.state_dict(),
    }

    loaded_model = torch.nn.Linear(2, 1)
    loaded_optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.1)

    checkpoint_path = tmp_path / "checkpoint.pth.tar"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_data = load_checkpoint(checkpoint_path, loaded_model, "cpu", loaded_optimizer)

    assert "state_dict" in checkpoint_data
    assert torch.equal(loaded_model.weight, model.weight)
    assert torch.equal(loaded_model.bias, model.bias)


def test_load_checkpoint_raises_for_missing_file():
    model = torch.nn.Linear(2, 1)

    with pytest.raises(FileNotFoundError):
        load_checkpoint("/tmp/does-not-exist-checkpoint.pth.tar", model, "cpu")


def test_load_checkpoint_raises_for_missing_state_dict(tmp_path):
    checkpoint_path = tmp_path / "broken-checkpoint.pth.tar"
    torch.save({"weights": {}}, checkpoint_path)
    model = torch.nn.Linear(2, 1)

    with pytest.raises(ValueError, match="state_dict"):
        load_checkpoint(checkpoint_path, model, "cpu")


def test_load_model_round_trip(tmp_path):
    checkpoint_path = tmp_path / "model.pth.tar"
    payload = {"weights": torch.tensor([1.0, 2.0])}
    torch.save(payload, checkpoint_path)

    loaded = load_model(checkpoint_path, "cpu")

    assert torch.equal(loaded["weights"], payload["weights"])


def test_parse_input_path_accepts_single_audio_file(tmp_path):
    audio_path = tmp_path / "mouse.wav"
    audio_path.write_bytes(b"audio")

    files = parse_input_path(audio_path)

    assert files == [str(audio_path)]


def test_parse_input_path_collects_supported_audio_extensions_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    wav_path = tmp_path / "nested" / "mouse.wav"
    flac_path = tmp_path / "rat.FLAC"
    wav_path.write_bytes(b"wav")
    flac_path.write_bytes(b"flac")

    files = parse_input_path(tmp_path, search_tree=True)

    assert files == sorted([str(wav_path), str(flac_path)])


def test_parse_input_path_raises_for_missing_path():
    with pytest.raises(InputPathError, match="usage: vocalpy"):
        parse_input_path()

    with pytest.raises(InputPathError, match="not a file or directory"):
        parse_input_path("/tmp/does-not-exist")


def test_create_output_directory_structure_uses_shared_output_path_helper():
    files = ["/tmp/mouse.wav", "/tmp/nested/rat.FLAC"]

    output_dirs = create_output_directory_structure(files)

    assert output_dirs == ["/tmp/mouse_outputs", "/tmp/nested/rat_outputs"]
    assert get_output_directory_for_audio_file("/tmp/mouse.wav") == "/tmp/mouse_outputs"


def test_create_directory_is_idempotent(tmp_path):
    path = tmp_path / "outputs" / "spectrogram"

    create_directory(path)
    create_directory(path)

    assert path.is_dir()


def test_remove_directory_removes_nested_tree(tmp_path):
    path = tmp_path / "outputs" / "spectrogram"
    path.mkdir(parents=True)
    (path / "frame.png").write_bytes(b"data")

    remove_directory(path.parent)

    assert not path.parent.exists()
