# Dev-only fixture generator (needs torch + official timestral; end users
# never run this). Creates:
#   tests/fixtures/tiny/           a tiny random-weight checkpoint
#   tests/fixtures/golden_*.csv    official-PyTorch outputs for that checkpoint
#   /tmp/timestral-synth-<name>/   config-variant checkpoints for step-2 checks
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from timestral.config import TimestralConfig
from timestral.model import TimestralModel
from timestral import TimestralForecaster

ROOT = Path(__file__).resolve().parent.parent
KNOTS = [0.01, 0.05] + [round(0.1 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99]


def make_ckpt(dirpath, seed, **overrides):
    cfg = dict(patch_size=16, d_model=32, num_heads=2, mlp_hidden_dim=64,
               block_schedule=["time", "time_variate", "variate_time"],
               quantile_knots=KNOTS, block_size=4, context_length=512,
               quantile_width_factor=3.0, attn_bias=True, mlp_bias=True)
    cfg.update(overrides)
    config = TimestralConfig(**cfg)
    torch.manual_seed(seed)
    model = TimestralModel(config).eval()
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    with open(dirpath / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    save_file(model.state_dict(), str(dirpath / "model.safetensors"),
              metadata={"format": "pt"})
    return dirpath


def main():
    tiny = make_ckpt(ROOT / "tests" / "fixtures" / "tiny", seed=123)
    f = TimestralForecaster.from_pretrained(tiny, max_context=4096)
    rng = np.random.default_rng(7)
    t = np.arange(96)
    y = 10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.2, 96)
    y[10:14] = np.nan
    np.savetxt(ROOT / "tests" / "fixtures" / "input_univariate.csv", y, delimiter=",")
    with torch.no_grad():
        fc = f.forecast(y, 24)
    np.savetxt(ROOT / "tests" / "fixtures" / "golden_univariate.csv",
               np.asarray(fc.quantiles)[0, 0], delimiter=",")

    # covariates case: 1 target + 1 past + 1 known covariate
    past = rng.normal(0, 1, 96)
    known = np.sin(2 * np.pi * np.arange(96 + 24) / 24)
    y2 = 20 + 2 * known[:96] + rng.normal(0, 0.2, 96)
    np.savetxt(ROOT / "tests" / "fixtures" / "input_cov_target.csv", y2, delimiter=",")
    np.savetxt(ROOT / "tests" / "fixtures" / "input_cov_past.csv", past, delimiter=",")
    np.savetxt(ROOT / "tests" / "fixtures" / "input_known.csv", known, delimiter=",")
    with torch.no_grad():
        fc = f.forecast(y2, 24, past_covariates=past[None, None],
                        known_covariates=known[None, None])
    np.savetxt(ROOT / "tests" / "fixtures" / "golden_covariates.csv",
               np.asarray(fc.quantiles)[0, 0], delimiter=",")

    # config variants for synthetic parity (not committed outputs; compared live)
    variants = {
        "bias": {},
        "nogate": {"time_attn_gate": "none", "variate_attn_gate": "none",
                   "attn_bias": False, "mlp_bias": False},
        "noasinh": {"use_asinh": False,
                    "block_schedule": ["variate_time", "time", "time_variate", "time"]},
    }
    for name, ov in variants.items():
        make_ckpt(f"/tmp/timestral-synth-{name}", seed=99, **ov)
    print("fixtures written")


if __name__ == "__main__":
    main()
