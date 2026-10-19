# timestralR 0.3.0

- Native RcppEigen backend (default, `backend = "native"` in
  `timestral_load()`): the single-block transformer primitive runs in
  C++ with Eigen, with large products routed to the system BLAS and
  small ones kept in Eigen (threshold tuned by benchmark; the tuned
  BLAS is not always faster). The pure-R implementation remains as
  `backend = "R"`, the reference and fallback. Results agree to < 1e-10
  on the golden fixtures; official parity is unchanged.
- Speed on this machine: 209M long-context case 13.8 s (R) -> 6.4 s
  (native); 5M 0.80 s -> 0.22 s.

# timestralR 0.2.0

- Complete parity with the Python `timestral` package: batched series
  (list or 3-D array input), `observation_mask`, categorical past/known
  covariates with target encoding, and the forecaster/loader options
  `per_shot_horizon_patches`, `quantile_real_cap_k`,
  `extreme_value_threshold`, `revision`, `cache_dir`, `local_files_only`.
- New accessors `timestral_median()` and `timestral_quantile()`.
- Config validation and strict weight checking at load, matching upstream;
  checkpoints with biases or non-default gates/schedules are supported.
- Breaking: `max_context` now defaults to the checkpoint's trained context
  length (32768), as in Python, instead of 4096. Pass
  `max_context = 4096L` to keep the previous behaviour.
- Golden-fixture tests (tiny checkpoint in `tests/fixtures/`) now run in
  every check, with no Python or downloads required.

# timestralR 0.1.0

- Initial native implementation: official fp32 safetensors checkpoints,
  forward pass and forecasting loop in pure R. Parity vs official PyTorch
  worst `max_rel` 2.0e-6 (5M, six cases).
