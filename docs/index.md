# VocalPy

VocalPy detects and classifies animal vocalizations from audio recordings.
The maintained package surface is centered on the packaged CLI, bundled
classifier checkpoints, optional neural vocal segmentation, and the shipped
mouse example fixtures.

## Start here

- Installation and packaging overview: [repository README](https://github.com/gumadeiras/vocalpy/blob/master/README.md)
- Maintained versus archived example notebooks: [examples guide](https://github.com/gumadeiras/vocalpy/blob/master/examples/README.md)
- Maintained example audio fixtures: [fixture notes](https://github.com/gumadeiras/vocalpy/blob/master/examples/audios/README.md)

## Operational checks

Validate bundled checkpoints:

```sh
vocalpy-models validate --smoke-test
```

Validate the maintained mouse baselines:

```sh
python scripts/compare_example_baselines.py --species mouse
```

`.vocalpy` artifacts now use a versioned envelope with object-type metadata.
Legacy raw-pickle `.vocalpy` files still load, but maintained fixtures are
expected to be rewritten through the current package helpers.

Optional neural vocal segmentation can be enabled from the CLI with:

```sh
vocalpy --path_to_audio /path/to/audio.wav --segmenter --segmentation_model_path /path/to/segmenter.pt
```

## API reference

- <a href="genindex.html">General index</a>
- <a href="py-modindex.html">Python module index</a>

```{toctree}
:hidden:

_source/vocalpy
```
