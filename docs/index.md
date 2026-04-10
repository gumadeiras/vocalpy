# VocalPy

VocalPy detects, classifies, and segments animal vocalizations from audio recordings. It ships a CLI backed by species-specific detection pipelines, bundled pretrained classifiers, and a neural segmentation stage (SqueakOut).

The pipeline runs in three stages: **detection** finds candidate vocalizations in the spectrogram using contrast- and morphology-based methods tuned per species; **classification** filters out noise and labels each remaining call by type using pretrained MobileNetV2 models; **segmentation** optionally runs SqueakOut to produce a pixel-level binary mask for each detected call. You can run detection alone, detection + classification, or all three.

**Supported species:** mouse, rat, guinea pig

## Start here

- [Getting started](getting-started.md) — installation, CLI reference, output format
- [Example fixtures](example-fixtures.md) — maintained audio fixtures and demo notebooks
- [Operational checks](operations.md) — model validation, baseline checks, docs deploy
- <a href="_source/vocalpy.html">API reference</a>

```{toctree}
:maxdepth: 2
:caption: Documentation

getting-started
example-fixtures
operations
_source/vocalpy
```
