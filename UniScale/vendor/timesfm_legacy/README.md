# TimesFM legacy PyTorch inference

This directory vendors the minimal official TimesFM PyTorch inference sources
needed by the TimesFM 1.0 and 2.0 checkpoints:

- upstream repository: `google-research/timesfm`
- upstream revision: `3dae50b20d7a724981e8ea36cda75578f80dd2dc`
- upstream location: `v1/src/timesfm`
- license: Apache-2.0, reproduced in `LICENSE`

The local copy retains these integration changes:

1. imports use the repository-local `UniScale.vendor.timesfm_legacy`
   namespace, avoiding any installed `timesfm` package;
2. unused DataFrame and covariate-regression convenience APIs, their helper
   functions, and their optional imports are omitted; the registered experiments
   use the array-based `forecast` path;
3. the abstract base implementation raises `NotImplementedError` explicitly.

Model architecture, checkpoint loading, preprocessing, normalization, decoding,
and quantile generation are otherwise kept from the cited upstream revision.
