# History-learning experiments

The scientific protocol is `history-learning-v5`, configured by
`UniScale/configs/learning_mechanism.json`. It covers synthetic context responses,
parameter interventions, activation transfer and recovery, and crossed-history
training on GIFT data.

## Experimental matrix

| Component | Tasks | Individual training fits |
| --- | ---: | ---: |
| Frozen context response: TimesFM 2.5 and Chronos 2, three processes | 6 | 0 |
| Parameter retention: two architectures, three processes, three lengths, three seeds | 54 | 108 |
| TimesFM activation grid: three processes and three sites | 9 | 0 |
| Chronos activation development, shared selection and confirmation | 13 | 0 |
| Crossed real-data controls: 23 tasks, four models, two histories, two inputs, three seeds | 1104 | 1104 |
| Total | 1186 | 1212 |

The matched-history full-shot comparison uses `UniScale/experiments/full_shot.py`
and `UniScale/configs/matched_history_full_shot.json`.

## Scientific protocol

Three stationary processes have opposite transition rules with identical
one-point marginals: Gaussian AR(1), Gaussian lag-8, and a nonlinear threshold
process. Magnitudes are 0.15, 0.225 and 0.30. A 96-observation query is sampled
from the stationary rule mixture and retained when its absolute rule
log-likelihood ratio is at most log(3). Each independent query has an antithetic
partner. Conditional historical prefixes preserve that current query. The primary
forecasting target is the one-step conditional mean; h2/h3 conditional-mean
errors are retained separately. Manuscript accuracy tables use MASE normalized
by the matched naive score. Conditional-mean MSE is retained for the exact
paired-rule decomposition and the original registered squared-loss endpoint.
All comparisons preserve paired queries and rule signs. Synthetic evaluation
archives contain queries and conditional means. Query, context, and training
generation use independently seeded random streams. Training uses observed
trajectory targets.

Frozen-model tests use context lengths 96, 128, 256, 512, 1024 and 2048, four
prefix repetitions, native median outputs and native or fixed population
normalization. First-block predictions use no rollout or cross-window cache.
Model identities and weight digests are pinned in `learning_mechanism_models.json`.

DLinear and compact PatchTST train on lengths 1024, 4096 and 8192, with seeds
2026, 2027 and 2028 and opposite-rule pairs. Training uses an 80% chronological
split, 2000 updates, batch 256 and validation-selected weights, without refitting.
PatchTST uses cosine scheduling and weight decay 0.01. Parameter exchanges copy
learned parameters and recalibrate BatchNorm on the same independent unlabelled
mixture for both directions: 512 windows per rule, length 96, batch 32. Calibration
uses no future targets or optimizer updates. DLinear has no BatchNorm and needs
no calibration forwards. All seeds, generators and schedules are in the config.

### Shared-site activation transfer and recovery

The primary TimesFM reporting setting uses residual block index 5 (zero-based), the final three query
tokens, 512 observations, strength 1, and four independent donor histories.
Each donor contributes its long-history state minus its own 96-step state;
these differences are averaged before the recipient forward pass. No
rule-labelled prototypes, fitted subspaces, parameter updates or future targets
enter this construction. Known synthetic rules determine matched/opposite donor
assignment and the scoring targets.

Transfer adds this difference to a short-query execution. Recovery replaces
the corresponding state in a long-context execution by its short-query state,
then adds the same donor difference; downstream layers still see the prefix.
All controls use the same donor count and strength: norm-matched random changes,
carrier-only differences, and shuffled-prefix histories with the last 96 values
unchanged. Natural predictions, erasure, random erasure, zero transfer and exact
restoration are also saved. Restoration identities are implementation checks.
The explicit donor information budget is four histories of 512 observations.

The primary TimesFM setting was fixed before the confirmation queries and is
read directly from the registered robustness grid without duplicate inference.
The activation query seed is 2026091701 and donor seed 2026091702. Each process/magnitude has 1024 independent
test queries and two donor repetitions. Repetitions share recipient queries and
are not independent query samples. No performance threshold determines which
results are retained.

### Crossed real-data controls

The same 23-task panel crosses training history T=4096/8192 with immediate input
C=96/512 at H=96 and shared origins registered with origin horizon 720. Models
are DLinear, PatchTST, compact PatchTST, and pretrained TimesFM 2.5. Each uses
three seeds, 400 optimizer updates, effective batch 32 and microbatch 8. Initial,
100-update, 400-update and validation-selected predictions are all retained;
selection includes the unmodified initial predictor. No final refit is performed.

Training starts and validation targets are constructed once at C=512 and then
inputs are cropped for C=96. Raw-history and evaluation-target digests must match
across arms. The full-shot normalization/loss configuration is pinned by hash,
including its registered high-volatility settings. Checkpoint artifacts are not written by the final protocol.

Every task retains GIFT point metrics (ND, MAE, MSE, RMSE, NRMSE and raw MASE),
the corresponding SeasonalNaive scores, and their model/baseline ratios in
`point_metrics.json`. Scoring calls the same `gluonts.ev` metric classes as GIFT.
The saved deterministic forecast is used as both mean and median; this adds no
artificial probabilistic output. Crossed tasks also retain per-variable/per-origin
additive losses in `point_statistics.npz`, including scaled loss sums and counts.
`training.json` records available history lengths for
the same evaluation origins.
MASE uses raw pre-origin histories with GIFT missing-value masking. SeasonalNaive
uses the registered seasonal period and causal last-value imputation. Synthetic
tasks use the common 96-step query for scaling and a lag-one naive forecast, with
conditional means scored separately at each horizon endpoint.
This preserves common scoring weights across paired interventions. The original
conditional-mean squared-error decomposition remains a separate mechanism endpoint.
All-missing histories follow GIFT's zero-valued imputation for the baseline;
their raw-history scaling errors remain masked. Stored observation counts make
the support of scaled and unscaled losses explicit.

## Run and resume

On a GPU host with the registered environments and an initial Git commit, use a new timestamp:

```bash
python -B -u -m UniScale.mechanism.pool \
  --run YYYYMMDDTHHMMSSffffff+HHMM \
  --model-root /path/to/models_hf --server-workspace /path/to/workspace \
  --gpu-ids 0 1 --workers-per-gpu 4 --maximum-attempts 2
```

The default config schedules the entire final matrix. `--task-kinds` can restrict
execution to `functional`, `weights`, `crossed_history`, or the registered
`specificity_*` kinds without changing those experiments. Chronos selection and
confirmation require their development dependencies. Resume with the same arguments and
`--resume`; configuration, Git revision and input hashes must match. GPU slots,
OOM batch reduction, task locks and retry handling apply to all runs. Runtime
files remain isolated in the new result directory. Existing results and source
model checkpoints are never overwritten. Hardware/runtime differences can affect
floating-point results; reproducibility means the registered scientific protocol,
data streams, task coverage and outputs, not guaranteed bitwise GPU equality.

### Rule specificity and cross-model confirmation

The default configuration includes all activation experiments as part of the
same native task matrix and evidence directory. No supplementary configuration,
legacy runner or second result directory is needed.

Its 22 tasks comprise nine TimesFM site tasks, nine Chronos development tasks,
one shared-setting selection task and three Chronos confirmation tasks. Each
site task reuses captured donor states across strengths 0.5, 1 and 1.5. TimesFM
tests zero-based blocks 3, 5 and 7 and retains all results. Chronos tests blocks
2, 5 and 8 (one quarter, half and three quarters of its twelve blocks). One
setting is selected across all development processes and both objectives using
conditional-mean normalized MASE, then fixed before independent confirmation.
All magnitudes and donor repetitions have equal weight. There is no test-based
selection or success threshold.

Both models replace the tokens corresponding exactly to the final 96 observed
values: three 32-step TimesFM patches or six 16-step Chronos patches. Chronos REG
and future tokens are not patched; unique group IDs prevent cross-query mixing.
An opposite-rule donor changes only the donor rule, keeping the recipient's
query and long history fixed. Donor carriers and sampling random numbers are
paired across signs. Random, shuffled-history, carrier-only, zero-change and
natural-restoration controls are retained. Queries and donors use fresh seeds;
development and confirmation donors use distinct streams.

All activation tasks store predictions, configuration, selection provenance and
the same GIFT point metrics and SeasonalNaive scores as the other experiments.
Development uses 128 independent queries at each magnitude and a separate donor
stream; confirmation uses 1024 independent queries at each magnitude.

The scoring modules and development-set selection are required during experiment
execution. Every run generates its own evidence, predictions,
metrics, and provenance under `UniScale/results/learning-mechanism/<timestamp>/`.

The pool requires two GPUs and a workspace containing this repository,
the model directory, and (for crossed-history tasks) `GiftEval_data/`. Supply
`--server-workspace` and `--model-root` for that workspace. Configure environment
interpreter paths as described in the root README. Full-shot/parameter training
uses `toto`; TSFM activation and TimesFM adaptation use `TSFM`.
