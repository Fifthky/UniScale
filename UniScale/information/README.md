# UniScale information catalogue

`models.json` is the machine-readable checkpoint catalogue used by the
experiments. It records model identities, capacities, context support, inference
settings, and provenance from model cards, upstream notebooks, and GIFT result
configurations.

The catalogue keeps checkpoint-level records independent. `verification_grade`
preserves uncertainty, `testdata_leakage` records the GIFT submission flag, and
`include_in_scaling` defines the default clean statistical panel. A missing
number is encoded as JSON `null`; it is never replaced by a guessed value.

Controlled experiment membership is defined separately.
`configs/formal_models.json` contains the 21-checkpoint set used by Controlled
GIFT-Horizon and Controlled Grid-Horizon. Public-result inclusion flags do not
determine controlled membership. The catalogue also records native rollout
behavior as execution provenance for discussion and future mechanism studies.

The Granite adapters load the pinned repository-local implementations in
`UniScale/vendor/flowstate` and `UniScale/vendor/patchtstfm`. Their source
revisions are recorded in `SOURCE.json`. Both use the
`granite` environment described in the root `ENVIRONMENTS.md`.

The GIFT column named `eval_metrics/mean_weighted_sum_quantile_loss` is called
CRPS throughout UniScale. Both CRPS and MASE are divided by the exact matching
seasonal-naive configuration before any statistical transformation.
