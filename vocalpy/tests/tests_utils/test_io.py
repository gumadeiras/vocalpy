# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import pytest
import torch

from vocalpy.utils.io import load_checkpoint


def test_save_file():
    assert True


def test_load_file():
    assert True


def test_load_recording_data():
    assert True


def test_load_checkpoint():
    assert True


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


def test_load_model():
    assert True


def test_parse_input_path():
    assert True


def test_create_output_directory_structure():
    assert True


def test_create_directory():
    assert True


def test_remove_directory():

    assert True
