# Operational checks

## Bundled models

Validate the shipped checkpoints and print their metadata:

```sh
vocalpy-models list
vocalpy-models validate --smoke-test
```

## Example baseline validation

Validate the maintained mouse fixtures against the current pipeline:

```sh
python scripts/compare_example_baselines.py --species mouse
```

## Docs deploy

Docs publish automatically from the `master` branch to
`https://vocalpy.gumadeiras.com` through GitHub Pages.

DNS needed:

```text
CNAME vocalpy -> gumadeiras.github.io
```

## Reference shortcuts

- <a href="genindex.html">General index</a>
- <a href="py-modindex.html">Python module index</a>
