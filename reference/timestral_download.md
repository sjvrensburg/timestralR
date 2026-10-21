# Download a Timestral checkpoint

Downloads `config.json` and `model.safetensors` from a Hugging Face Hub
repository, cached per user. A size names the official Mistral AI
repository (Modified MIT weights licence); any other `"owner/name"`
repository holding a Timestral checkpoint in the same format (for
example a fine-tuned model) can be given by its ID.

## Usage

``` r
timestral_download(
  model = "5M",
  revision = NULL,
  cache_dir = tools::R_user_dir("timestralR", "cache"),
  local_files_only = FALSE,
  token = Sys.getenv("HF_TOKEN", "")
)
```

## Arguments

- model:

  one of "5M", "28M", "123M", "209M", or a Hub repository ID such as
  `"mistralai/Timestral-5M"`. The official repositories are pinned
  whichever way they are named.

- revision:

  Hub revision (branch, tag or commit). The default `NULL` uses the
  pinned upstream commit for an official size, whose files are verified
  against recorded SHA-256 hashes, and `"main"` for any other
  repository. Branch and tag names are resolved to their current commit
  on the Hub (the resolution is cached for offline use); files at
  revisions other than the pinned ones are not hash-verified.

- cache_dir:

  where to keep the files

- local_files_only:

  only read from the cache, never download

- token:

  Hugging Face access token for private or gated repositories; defaults
  to the `HF_TOKEN` environment variable

## Value

The checkpoint directory.

## Examples

``` r
if (FALSE) { # \dontrun{
timestral_download("5M")
timestral_download("mistralai/Timestral-5M")   # the same, pinned
} # }
```
