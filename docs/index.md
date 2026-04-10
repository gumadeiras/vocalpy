# VocalPy

## Bundled models

Bundled classifier checkpoints include sidecar metadata with architecture, classes,
input shape, and SHA256 hashes. Validate them with:

```sh
vocalpy-models validate --smoke-test
```

## Serialized outputs

`.vocalpy` artifacts now use a versioned envelope with object-type metadata so
`recording` and `list_of_vocals` payloads can be validated on load. Legacy raw
pickle `.vocalpy` files still load, but maintained fixtures should be rewritten
through the current package helpers.

## Example baselines

Shipped example baseline fixtures are declared in `examples/audios/baselines.yml`.
The current repo only ships mouse audio, so rat and guinea pig baseline support
is implemented through the shared manifest-driven tooling and becomes active as
soon as those fixtures are added.

```sh
python scripts/compare_example_baselines.py --species mouse
```
![Python Versions](https://img.shields.io/badge/python-3.12-blue)
![Platforms](https://img.shields.io/badge/platform-linux--64%20%7C%20osx--64-lightgrey)
![build](https://github.com/gumadeiras/vocalpy/workflows/build/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/gumadeiras/vocalpy/branch/master/graph/badge.svg?token=vBVu77sJ5R)](https://codecov.io/gh/gumadeiras/vocalpy)
[![CodeFactor](https://www.codefactor.io/repository/github/gumadeiras/vocalpy/badge?s=e1ba6c8796b9923a3cdcfd1e51fcf368a743ab83)](https://www.codefactor.io/repository/github/gumadeiras/vocalpy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
![License](https://img.shields.io/badge/license-Apache%202-blue)


# installation

## quickstart (recommended)

It is not required, but **highly recommended** to install using a virtual environment. The supported runtime is Python `3.12`.

For `micromamba`:
```sh
micromamba create -y -n vocalpy python=3.12
micromamba activate vocalpy
git clone https://github.com/gumadeiras/vocalpy.git
cd vocalpy
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## direct from source

To clone the repository locally and install in editable mode run

```sh
git clone https://github.com/gumadeiras/vocalpy.git
cd vocalpy
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

# code documentation

- <a href="genindex.html">Index</a>
- <a href="py-modindex.html">Python Module Index</a>
