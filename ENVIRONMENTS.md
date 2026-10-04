# Environment versions

Use Python 3.12 with PyTorch 2.8.0a0 and CUDA 12.9. The table lists the main
package versions used by each experiment environment.

| Package | TSFM / granite | tirex2 | toto |
| --- | --- | --- | --- |
| NumPy | 1.26.4 | 1.26.4 | 1.26.4 |
| Pandas | 2.1.4 | 2.2.3 | 2.2.3 |
| SciPy | 1.11.4 | 1.15.3 | 1.15.3 |
| GluonTS | 0.15.1 | 0.17.0 | 0.16.2 |
| datasets | 2.17.1 | 3.6.0 | 2.17.1 |
| PyArrow | 23.0.0 | 21.0.0 | 21.0.0 |

Model packages:

- `TSFM`: chronos-forecasting 2.2.2, uni2ts 2.0.0, tirex-ts 1.4.0.
- `granite`: FlowState and PatchTST-FM implementations in `UniScale/vendor/`.
- `tirex2`: tirex-2 0.1.1, flashrnn 1.0.6, mlstm-kernels 2.0.4, xlstm 2.0.5.
- `toto`: toto-2 2.0.0; also used for DLinear and PatchTST training.

TimesFM implementations are included in `UniScale/vendor/`. Configure
interpreter and model paths as described in the root README.
