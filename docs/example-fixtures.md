# Example fixtures

## Maintained audio fixtures

`mouse_1.wav` and `mouse_2.wav` plus their matching `*_outputs/` directories
are the maintained example fixtures. They back the exact baseline smoke checks
in CI.

- canonical manifest: `examples/audios/baselines.yml`
- refresh helper: `python scripts/rewrite_example_vocalpy_fixtures.py`
- bundled neural masks: `examples/audios/mouse_1_outputs/cnn_mask/` and `examples/audios/mouse_2_outputs/cnn_mask/`

## Maintained notebooks

- `class_classifier_demo.ipynb`
- `noise_classifier_demo.ipynb`
- `visualization_plots_demo.ipynb`

## Archived notebooks

- `candidate_vocalization_identifier_demo.ipynb`
- `unsupervised_embedding.ipynb`
- `unsupervised_embedding-Copy1.ipynb`

Archived notebooks stay for historical context only. They are not the supported
public surface.

## Legacy demo data

`examples/audios/example_outputs/` is legacy image-only classifier demo data.
It is not part of the CI baseline manifest and should not be treated as modern
canonical pipeline output.
