# VocalPy
![Python Versions](https://img.shields.io/badge/python-3.12-blue)
![Platforms](https://img.shields.io/badge/platform-linux--64%20%7C%20osx--64-lightgrey)
![build](https://github.com/gumadeiras/vocalpy/workflows/build/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/gumadeiras/vocalpy/branch/master/graph/badge.svg?token=vBVu77sJ5R)](https://codecov.io/gh/gumadeiras/vocalpy)
[![CodeFactor](https://www.codefactor.io/repository/github/gumadeiras/vocalpy/badge?s=e1ba6c8796b9923a3cdcfd1e51fcf368a743ab83)](https://www.codefactor.io/repository/github/gumadeiras/vocalpy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
![License](https://img.shields.io/badge/license-Apache%202-blue)


## Installation

#### Preliminaries

- Make sure you have installed [Git LFS](https://git-lfs.github.com/).

- It is not required, but **highly recommended** to install using a virtual environment.
- The supported runtime is Python `3.12`.

#### Quickstart with `micromamba` (recommended)

```sh
micromamba create -y -n vocalpy python=3.12
micromamba activate vocalpy
git clone https://github.com/gumadeiras/vocalpy.git
cd vocalpy
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

#### Quickstart with `venv`

```sh
python3.12 -m venv .venv
source .venv/bin/activate
git clone https://github.com/gumadeiras/vocalpy.git
cd vocalpy
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

#### Direct from source (not recommended)

To clone the repository and install only the runtime dependencies:
```sh
git clone https://github.com/gumadeiras/vocalpy.git
cd vocalpy
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### Packaging notes

- Project metadata now lives in `pyproject.toml`.
- Constraints for the tested dependency set live in `constraints/base.txt` and `constraints/dev.txt`.
- Bundled classifier checkpoints ship with sidecar metadata in `vocalpy/nn/pretrained/*.metadata.json`.

#### Bundled model validation

Validate the shipped checkpoints and print their metadata:

```sh
vocalpy-models list
vocalpy-models validate --smoke-test
```
