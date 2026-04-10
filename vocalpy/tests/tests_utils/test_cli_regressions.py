# -*- coding: utf-8 -*-
"""Regression tests for CLI argument parsing."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from vocalpy.cli import build_parser


def test_cli_uses_default_frequency_sentinels_without_parse_failure():
    args = build_parser().parse_args([])

    assert args.lower_frequency_cutoff == "default"
    assert args.higher_frequency_cutoff == "default"


def test_cli_parses_numeric_frequency_overrides_as_integers():
    args = build_parser().parse_args(["-lf", "45000", "-hf", "125000"])

    assert args.lower_frequency_cutoff == 45000
    assert args.higher_frequency_cutoff == 125000
