# Load a Timestral model

Load a Timestral model from the official Mistral AI checkpoint, another
Hugging Face Hub repository, or a local checkpoint directory.

## Usage

``` r
timestral_load(
  model = "5M",
  max_context = NULL,
  per_shot_horizon_patches = 32L,
  quantile_real_cap_k = 10000,
  extreme_value_threshold = 1e+30,
  revision = NULL,
  cache_dir = tools::R_user_dir("timestralR", "cache"),
  local_files_only = FALSE,
  token = Sys.getenv("HF_TOKEN", ""),
  backend = c("native", "R")
)
```

## Arguments

- model:

  a size ("5M", "28M", "123M", "209M") to download/cache, a local
  directory holding `config.json` and `model.safetensors`, or a Hugging
  Face repository ID (`"owner/name"`) to download/cache. A local
  directory takes precedence over a repository ID of the same name.

- max_context:

  context cap in steps; NULL (default) uses the trained context length
  from the checkpoint, as in the Python package

- per_shot_horizon_patches:

  patches decoded per block (capped at the trained block size)

- quantile_real_cap_k:

  clamp forecasts to the observed context range plus/minus this multiple
  of the last patch scale; 0 disables

- extreme_value_threshold:

  fed-back medians at or above this magnitude become 0

- revision:

  Hub revision when `model` is a size or repository ID; see
  [`timestral_download()`](https://sjvrensburg.github.io/timestralR/reference/timestral_download.md).
  The default `NULL` uses the pinned, hash-verified upstream commit for
  an official size and `"main"` otherwise.

- cache_dir:

  checkpoint cache directory for downloaded checkpoints

- local_files_only:

  only read from the cache, never download

- token:

  Hugging Face access token for private or gated repositories; defaults
  to the `HF_TOKEN` environment variable

- backend:

  "native" (default) uses the compiled RcppEigen implementation of the
  transformer block; "R" uses the pure-R reference implementation.
  Results are identical up to floating-point order.

## Value

A `timestral_model` object.

## Examples

``` r
## A tiny random-weight checkpoint ships with the package, so this runs
## offline; its forecasts are meaningless. Use a size such as "5M" or
## "28M" (downloaded once) for real forecasts.
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
fc <- timestral_forecast(m, AirPassengers, h = 24)
plot(fc)

if (FALSE) { # \dontrun{
m <- timestral_load("5M")
m <- timestral_load("mistralai/Timestral-28M")
} # }
```
