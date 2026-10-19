# Dev-only reference: official Timestral (PyTorch) forecast for one case.
# Usage: official_reference.py <model_dir> <case.json> <out.txt>
import sys, json, numpy as np, torch
from timestral import TimestralForecaster

model_dir, case, out = sys.argv[1:4]
d = json.load(open(case))

def arr(key):
    if d.get(key) is None:
        return None
    return np.array(d[key], dtype=float).reshape(d[key + "_dim"], order="F")[None]

torch.set_num_threads(2)
f = TimestralForecaster.from_pretrained(model_dir, max_context=4096)
with torch.no_grad():
    fc = f.forecast(arr("target"), d["h"], past_covariates=arr("past"),
                    known_covariates=arr("known"),
                    symmetric_averaging=d["sym"], clip_nonnegative=True)
q = np.asarray(fc.quantiles)[0].astype(np.float64)
np.savetxt(out, q.flatten(order="F"))
