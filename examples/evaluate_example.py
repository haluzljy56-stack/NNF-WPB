"""Read one NNF-WPB instance, decode a random plan and a greedy plan, and print their objectives."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nnfwpb import WPBProblem  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "instances")
code = sys.argv[1] if len(sys.argv) > 1 else "N48-S-A-N-08"
folder = sys.argv[2] if len(sys.argv) > 2 else "test"
prob = WPBProblem(os.path.join(ROOT, folder, code + ".json.xz"),
                  os.path.join(ROOT, folder, code + ".task.json.xz"))
print("%s: %d threats, %d candidate chains, %d ships" % (code, prob.n, prob.n_opps, len(prob.P.ship_ids)))
print("carrier survival floor:", prob.floor)

# 1) random integer chromosome
rng = np.random.default_rng(0)
x = np.array([rng.integers(0, u + 1) for u in prob.int_ub])
r = prob.decode_int(x)
print("random plan : O1 = %.4f  O2 = %.0f  floor violation = %.4f" % (r["O1"], r["O2"], r["CV"]))

# 2) every threat requests its first candidate chain, CIWS on where available
x = np.zeros(3 * prob.n, dtype=int)
for i in range(prob.n):
    x[3 * i] = 1 if prob.int_ub[3 * i] >= 1 else 0
    x[3 * i + 2] = prob.int_ub[3 * i + 2]
r = prob.decode_int(x)
print("greedy plan : O1 = %.4f  O2 = %.0f  floor violation = %.4f" % (r["O1"], r["O2"], r["CV"]))
print("total ship value:", prob.V_total)
