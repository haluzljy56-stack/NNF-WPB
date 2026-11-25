"""Problem interface for chromosome-based algorithms (the shared decoder of the paper, Section S-D3).

Objectives O1 (eq. (2)) and O2 (eq. (4)), constraints (5)-(9).

Decoding of a chromosome:
  * genes are inserted in the order (threat impact time, slot 1, slot 2); an infeasible gene is dropped
    and reported as a conflict;
  * providers are taken in a fixed order: the shooter itself and airborne platforms first, then by asset id;
  * one full evaluation of O1 per decoded plan.

Encodings
  int    : x of length 3n, per threat i (in index order) [slot1, slot2, pd]; slot value 0 = empty,
           k = k-th candidate of threat i (1-based, candidates in opportunity-index order); pd in {0,1}
  binary : b of length |opps| + n; b[e] = 1 selects opportunity e, b[|opps| + i] = 1 switches on PD of i
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .loader import load as _load, Problem, carrier_floor
from .decoder import Decoder, d1_compatible
from .engine import Engine, Plan


class WPBProblem:
    def __init__(self, core_path: str, task_path: str):
        self.P: Problem = _load(core_path, task_path, objective='v11', o2_mode='v11')
        P = self.P
        I = P.inst
        self.n = len(I.threats)
        self.cands: List[List[int]] = [sorted(I.chains_of[i]) for i in range(self.n)]
        self.order = list(I.order)                      # threats by impact time
        self.n_opps = len(P.opps)
        self.has_pd = np.array([1 if i in P.pd else 0 for i in range(self.n)])
        self.V_total = sum(s.V for s in I.ships)
        # fixed provider order
        for op in P.opps:
            for sv in op.services:
                sv.cands.sort(key=lambda c: (0 if (c.platform == op.shooter or c.platform not in P.ship_idx) else 1,
                                             c.asset))
        self.engine = Engine(I)
        self.n_decodes = 0
        # carrier survival floor (9): violation CV = max(0, floor - S_CV) of the last decode
        self.floor = carrier_floor(P)
        pid = next((p['platform_id'] for p in P.core['platforms'] if p.get('platform_role') == 'CARRIER'), None)
        self.cvi = P.ship_idx[pid] if pid is not None else None
        self.last_scv, self.last_cv = 1.0, 0.0
        # upper bounds for the integer encoding
        ub = []
        for i in range(self.n):
            ub += [len(self.cands[i]), len(self.cands[i]), int(self.has_pd[i])]
        self.int_ub = np.array(ub, dtype=int)

    # ------------------------------------------------------------------ core decoder
    def _decode_genes(self, rows: Dict[int, List[int]], pd_on: Dict[int, int], budget: Optional[float],
                      slot_major: bool = False):
        """rows[i] = candidate opportunity ids requested for threat i (in priority order).
        Gene order: target-major (threat by impact time, then slot 1, slot 2) by default; slot_major=True
        processes slot 1 of all threats first, then slot 2.
        conflicts: (slot, threat index, reason, position of the threat in the decoding order)."""
        P = self.P
        D = Decoder(P)
        o2 = 0.0
        conflicts = []
        kept = {i: [] for i in range(self.n)}
        pos = {i: k for k, i in enumerate(self.order)}
        if slot_major:
            seq = [(i, s) for s in (0, 1) for i in self.order]
        else:
            seq = [(i, s) for i in self.order for s in (0, 1)]
        for i, slot in seq:
            req = rows.get(i, [])
            if slot >= len(req) or req[slot] is None:      # None = explicitly empty slot
                continue
            e = req[slot]
            keep = kept[i]
            if True:
                if e in keep:
                    conflicts.append((slot, i, 'duplicate', pos[i])); continue
                op = P.opps[e]
                if budget is not None and o2 + op.cost > budget + 1e-9:
                    conflicts.append((slot, i, 'budget', pos[i])); continue
                if not D.can_engage(e):
                    if D.count[op.tgt] >= 2:
                        why = 'count'
                    elif D.used[(op.shooter, op.weapon)] + 1 > P.inventory.get((op.shooter, op.weapon), 0):
                        why = 'magazine'
                    else:
                        why = 'timing'
                    conflicts.append((slot, i, why, pos[i])); continue
                spec = D.probe(e)
                if spec is None:
                    conflicts.append((slot, i, 'capacity', pos[i])); continue
                D.commit(e, spec); o2 += op.cost; keep.append(e)
        for i in self.order:
            if pd_on.get(i) and i in P.pd:
                if D.can_pd(i):
                    D.add_pd(i)
                else:
                    conflicts.append((2, i, 'pd', pos[i]))
        # full evaluation of O1
        plan = Plan(self.n)
        for e, spec in D.sel.items():
            i = P.opps[e].tgt
            plan.X[i] = tuple(sorted(plan.X[i] + (e,)))
            plan.prov[e] = frozenset(P.ship_idx[p] for p in spec["providers"] if p in P.ship_idx)
        for i in D.pd_sel:
            plan.y[i] = 1
        O1 = self.engine.evaluate(plan)
        self.n_decodes += 1
        if self.cvi is not None:
            self.last_scv = self.engine.Ssum[self.cvi][-1]
        self.last_cv = max(0.0, self.floor - self.last_scv) if self.floor is not None else 0.0
        return D, kept, O1, o2, conflicts

    # ------------------------------------------------------------------ integer encoding
    def decode_int(self, x, budget: Optional[float] = None):
        x = np.asarray(x, dtype=int)
        rows, pd_on = {}, {}
        for i in range(self.n):
            req = []
            for s in (0, 1):
                k = int(x[3 * i + s])
                if 1 <= k <= len(self.cands[i]):
                    req.append(self.cands[i][k - 1])
            rows[i] = req
            pd_on[i] = int(x[3 * i + 2])
        D, kept, O1, O2, conf = self._decode_genes(rows, pd_on, budget)
        xr = np.zeros_like(x)
        for i in range(self.n):
            ks = [self.cands[i].index(e) + 1 for e in kept.get(i, [])]
            # keep each surviving gene in its original slot position where possible
            orig = [int(x[3 * i]), int(x[3 * i + 1])]
            out = [0, 0]
            for k in ks:
                if k in orig and out[orig.index(k)] == 0:
                    out[orig.index(k)] = k
                else:
                    out[out.index(0)] = k
            xr[3 * i], xr[3 * i + 1] = out
            xr[3 * i + 2] = 1 if i in D.pd_sel else 0
        return dict(x=xr, O1=O1, O2=O2, conflicts=conf, decoder=D, CV=self.last_cv, SCV=self.last_scv)

    # ------------------------------------------------------------------ binary encoding
    def decode_binary(self, b, budget: Optional[float] = None):
        b = np.asarray(b).astype(bool)
        rows, pd_on = {}, {}
        sel = np.flatnonzero(b[: self.n_opps])
        for e in sel:
            rows.setdefault(self.P.opps[e].tgt, []).append(int(e))
        for i in range(self.n):
            pd_on[i] = int(b[self.n_opps + i])
        D, kept, O1, O2, conf = self._decode_genes(rows, pd_on, budget)
        br = np.zeros(self.n_opps + self.n, dtype=bool)
        for i, ks in kept.items():
            br[ks] = True
        for i in D.pd_sel:
            br[self.n_opps + i] = True
        return dict(x=br, O1=O1, O2=O2, conflicts=conf, decoder=D, CV=self.last_cv, SCV=self.last_scv)


def penalised_f1(f1: float, cv: float) -> float:
    """scalar feasibility-first penalty for algorithms without native constraint handling:
    any infeasible plan is worse than every feasible one (feasible f1 <= 1), and less violation is better."""
    return f1 if cv <= 1e-12 else 2.0 + cv


class Archive:
    """all-time non-dominated archive over integer O2 levels (O1 max, O2 min), with time checkpoints.
    All methods in the paper report this all-time archive as their front."""

    def __init__(self, checkpoint: float = 2.0):
        self.A: Dict[int, Tuple[float, object]] = {}
        self.t0 = time.perf_counter()
        self.checkpoint = checkpoint
        self.trace = []

    def offer(self, O1: float, O2: float, payload=None, cv: float = 0.0) -> bool:
        if cv > 1e-12:                       # only plans that meet the carrier floor enter the archive
            self._tick(); return False
        lv = int(round(O2))
        for l, (v, _) in self.A.items():
            if l <= lv and v >= O1 - 1e-12:
                self._tick(); return False
        for l in [l for l, (v, _) in self.A.items() if l >= lv and v <= O1]:
            del self.A[l]
        self.A[lv] = (O1, payload)
        self._tick()
        return True

    def _tick(self, force=False):
        t = time.perf_counter() - self.t0
        if force or not self.trace or t - self.trace[-1][0] >= self.checkpoint:
            self.trace.append((round(t, 2), sorted((l, round(v, 6)) for l, (v, _) in self.A.items())))

    def front(self):
        return sorted((l, v) for l, (v, _) in self.A.items())
