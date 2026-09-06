# Third-party notices

## Ultralytics

`instrument_frozen/ultralytics_local/ultralytics/` is the frozen modified detector source and reports version 8.4.51. Its source headers identify the GNU Affero General Public License v3.0. Upstream: <https://github.com/ultralytics/ultralytics>. The upstream sample images `bus.jpg` and `zidane.jpg` are deliberately omitted.

## Datasets

- AI-TOD-v2: <https://github.com/Chasel-Tsui/AI-TOD-v2>
- VisDrone: <https://github.com/VisDrone/VisDrone-Dataset>

No dataset pixels or annotations are included. Image identifiers and inclusion probabilities are factual audit metadata and do not substitute for access under the dataset owners' terms.

## Runtime dependencies

PyTorch, NumPy, pandas, SciPy, Matplotlib, statsmodels and other packages are dependencies rather than redistributed source. Their own licences continue to apply. Review their notices when producing a binary/container distribution.

## Model weights

The separately staged checkpoints are author-trained artifacts based on the Ultralytics model implementation. All authors and the relevant institutional rights holder approved their public distribution as v3.1.0 release assets under AGPL-3.0-only.
