# Dev-only reference: official Timestral (PyTorch) forecast for one case.
# Usage: official_reference.py <model_dir> <case.json> <out.txt>
# Case JSON arrays are flattened in Fortran order with a "<key>_dim" shape.
import sys, json, numpy as np, torch
from timestral import TimestralForecaster

model_dir, case, out = sys.argv[1:4]
d = json.load(open(case))

def arr(key):
    if d.get(key) is None:
        return None
    return np.array(d[key], dtype=float).reshape(d[key + "_dim"], order="F")

torch.set_num_threads(2)
f = TimestralForecaster.from_pretrained(
    model_dir, max_context=4096,
    per_shot_horizon_patches=d.get("per_shot", 32),
    quantile_real_cap_k=d.get("cap_k", 1e4))
if "extreme" in d:
    f.extreme_value_threshold = d["extreme"]
with torch.no_grad():
    fc = f.forecast(arr("target"), d["h"], observation_mask=arr("obs_mask"),
                    past_covariates=arr("past"), known_covariates=arr("known"),
                    categorical_past_covariates=arr("cat_past"),
                    categorical_known_covariates=arr("cat_known"),
                    symmetric_averaging=d["sym"], clip_nonnegative=True)
q = np.asarray(fc.quantiles).astype(np.float64)  # (B, V, H, Q)
np.savetxt(out, q.flatten(order="F"))
