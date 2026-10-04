# Time-Series-Library model subset

This directory contains the DLinear and PatchTST long-term forecasting paths
adapted from [THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library).
The source project is distributed under the bundled MIT license.

The adaptation removes unrelated tasks and data loaders, retains the source
forecast computations, and exposes a strictly univariate constructor used by
the UniScale dataset-level matched-history experiment. One model shares its
parameters across every scalar series in a dataset. Training and evaluation
policy live in `UniScale/experiments/full_shot.py` and
`UniScale/experiments/matched_history.py`.
