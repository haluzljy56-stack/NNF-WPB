# NNF-WPB: Networked Naval Formation Wave Planning Benchmark

This repository contains the benchmark instances, the reading and evaluation code, and the
supplementary material of the paper *Structural-Knowledge-Guided Pareto Optimization for
Fire–Support Planning of Networked Naval Formations* (under review).

NNF-WPB describes the defense of a naval formation against one wave of a saturation attack. A
plan decides which ship engages which threat with which interceptor and when, which platforms
provide tracking, computation, uplink, relay, and illumination for each engagement, and which
threats the close-in weapon system (CIWS) of each ship engages. The two objectives are the
expected formation survival value O1 (maximized) and the interceptor expenditure O2
(minimized), subject to resource capacities, inventories, a self-defense reservation, at most
two engagements per threat, and a survival floor for the carrier. The model is given in
Section II of the paper and in Sections S-A and S-B of the supplementary material.

All platform, weapon, and threat types are generic categories. Parameter ranges are taken from
public sources where available and are otherwise synthetic values for sensitivity analysis;
the instances make no claim about the performance of any real system.

## Contents

```
instances/
  test/     224 instances of the main comparison (groups N48, N96, N144, XL-R15, XL-R10)
  ood/      18 out-of-distribution instances (9 pairs of in-distribution and shifted instances)
  factor/   8 instances with scenario factors outside the main design
  stress/   8 larger instances (N192, N288)
  n12/      15 instances with 12 threats used for the comparison with exact solutions
metadata/
  instances.csv   one row per instance (code, folder, group, factors, number of threats,
                  number of candidate chains where recorded, file size, SHA-256)
nnfwpb/           Python code that reads an instance and evaluates plans
examples/
  evaluate_example.py
supplementary/
  supplementary.pdf
```

Each instance consists of two xz-compressed JSON files:

* `<code>.json.xz` — the scenario: platforms and their positions, resource pools with
  capacities per time slot, communication links, inventories, threats with trajectories and
  impact times, engagement opportunities (candidate chains) with launch and intercept times,
  kill probabilities, and the providers qualified for each support service, and CIWS
  opportunities;
* `<code>.task.json.xz` — the task: ship values and hits to disable, the hit probability of a
  leaking threat, interceptor costs, the penalty for deviations from the zones of
  responsibility, rule switches, and the carrier survival floor (`kappa` and the reference
  value `smax_cv`).

The files can be read with any JSON library after xz decompression (for example,
`lzma.open` in Python).

## Instance codes

| Folder | Code | Example |
|---|---|---|
| test, n12 | group–arrival–relay–pressure–seed | `N96-P-A-B-11` |
| test (XL) | group–N–arrival–relay–pressure–seed | `XL-R15-209-S-A-N-19` |
| ood | `OOD`–shift–size code–`ID`/`OOD` | `OOD-THREATMIX-N96-S-A-N-00-ID` |
| factor | `FACTOR`–factor–size code–level | `FACTOR-GEOMETRY-N96-S-A-N-04-COMPACT` |
| stress | `SCALE`–size code | `SCALE-N192-S-A-N-00` |

Arrival: `S` salvo, `P` spread. Relay: `A` air, `S` surface. Pressure: `N` nominal, `I` lean
inventory, `S` lean support, `B` both lean. XL groups: inventory-to-threat ratio 1.5 (`R15`) or
1.0 (`R10`); `N` is the number of threats.

## Reading and evaluating an instance

Requirements: Python 3.9 or later and NumPy.

```python
from nnfwpb import WPBProblem

prob = WPBProblem("instances/test/N48-S-A-N-08.json.xz",
                  "instances/test/N48-S-A-N-08.task.json.xz")
print(prob.n, "threats,", prob.n_opps, "candidate chains")

# integer encoding: per threat [chain slot 1, chain slot 2, CIWS]; 0 = empty,
# k = k-th candidate chain of the threat
import numpy as np
x = np.zeros(3 * prob.n, dtype=int)
res = prob.decode_int(x)
print(res["O1"], res["O2"], res["CV"])   # survival value, expenditure, floor violation
```

`decode_int` and `decode_binary` insert the requested chains in the order of impact time,
drop every chain that would violate a constraint, and evaluate the resulting feasible plan
with the recursion (3) of the paper. `res["CV"] > 0` means that the plan does not meet the
carrier survival floor. `examples/evaluate_example.py` gives a complete example.

For the study with 12 candidate chains per threat (folder `n12`), load the instance with
`nnfwpb.load(scenario, task, objective="v11", o2_mode="v11", max_opps_per_target=12)`.

## Settings used in the paper

* Time limit per run: `T = 0.625 N` seconds.
* Carrier survival floor: `kappa = 0.8` (as stored in the task files).
* Expenditure limits of the single-objective track: `b2` = highest level of the deterministic
  completion trajectory of SKPO from the empty plan, `b1 = round(0.5 * b2)`.
* HV and IGD+ are computed after normalizing (−O1, O2) by the ideal and nadir points of the
  reference set of each instance, with the reference point (1.1, 1.1).

Further details are given in Section S-D of the supplementary material.

## License

Instances and supplementary material: CC BY 4.0. Code: MIT License. See `LICENSE`.
