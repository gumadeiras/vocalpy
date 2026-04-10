# Getting started

## Installation

Recommended:

```sh
micromamba create -y -n vocalpy python=3.12
micromamba activate vocalpy
git clone https://github.com/gumadeiras/vocalpy.git
cd vocalpy
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Alternative with `venv`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
git clone https://github.com/gumadeiras/vocalpy.git
cd vocalpy
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Packaging notes

- Project metadata lives in `pyproject.toml`.
- Tested dependency pins live in `constraints/base.txt` and `constraints/dev.txt`.
- Bundled model metadata lives beside the packaged checkpoints under `vocalpy/nn/pretrained/`.

## Serialized outputs

- `.vocalpy` files use a versioned envelope with object-type metadata.
- Legacy raw-pickle `.vocalpy` files still load for backward compatibility.
- Maintained example recordings are expected to be rewritten through package helpers, not saved as raw pickles.

## Neural segmentation

Run the packaged CLI with the bundled SqueakOut segmenter:

```sh
vocalpy --path_to_audio /path/to/audio.wav --segmenter
```

- Default checkpoint: bundled `SqueakOut`
- Override path: `--segmentation_model_path /path/to/squeakout_checkpoint.ckpt`
- Input: per-vocal spectrogram crops, resized to grayscale `1x512x512`
- Output: binary masks saved under `cnn_mask/`
- Default threshold: `0.51`
