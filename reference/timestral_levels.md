# Quantile levels of the official Timestral checkpoints

The levels every official checkpoint forecasts. A forecast always
carries the levels of the checkpoint that made it in `fc$levels`, which
can differ for other (for example fine-tuned) checkpoints.

## Usage

``` r
timestral_levels()
```

## Value

Numeric vector of 21 levels from 0.01 to 0.99.

## Examples

``` r
timestral_levels()
#>  [1] 0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70
#> [16] 0.75 0.80 0.85 0.90 0.95 0.99
```
