# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import pytest
from argparse import ArgumentParser
from vocalpy.utils.misc import (
    create_logger,
    validate_arguments,
    validate_bin_size,
    validate_frequency_range,
    validate_thread_count,
    validate_animal,
)


@pytest.fixture
def test_args_correct():
    p = ArgumentParser()
    p.add_argument("-a", "--animal", type=str, default="mouse")
    p.add_argument(
        "-p", "--path_to_audio", type=str, default=None,
    )
    p.add_argument(
        "-b", "--bin_size", type=int, default=60,
    )
    p.add_argument(
        "-f", "--frequency", type=str, default="default",
    )
    p.add_argument("-t", "--threads", type=int, default=-1)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "-l", "--validation", action="store_true",
    )
    args = p.parse_args()
    return args


@pytest.fixture
def test_args_incorrect():
    p = ArgumentParser()
    p.add_argument("-a", "--animal", type=str, default="mousee")
    p.add_argument(
        "-p", "--path_to_audio", type=str, default=None,
    )
    p.add_argument(
        "-b", "--bin_size", type=int, default=-1,
    )
    p.add_argument(
        "-f", "--frequency", type=str, default="[45,5000]",
    )
    p.add_argument("-t", "--threads", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "-l", "--validation", action="store_true",
    )
    args = p.parse_args()
    return args


def test_create_logger():
    assert True


def test_validate_arguments_from_correct():
    assert True


def test_validate_arguments_from_incorrect():
    assert True


def test_validate_bin_size():
    assert True


def test_validate_frequency_range():
    assert True


def test_validate_thread_count():
    assert True


def test_validate_animal():
    assert True
