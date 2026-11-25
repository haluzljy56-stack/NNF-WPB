"""Support-layer decoder with exact incremental feasibility (insertion and removal of chains).

Slot-based occupancy (Section S-B), one provider per service window, first fitting communication
path, launcher pool at the launch slot. Removal is exact, so every move costs O(slots touched).
The self-defense tracking reservation (6) counts the tracking use for threats not aimed at the
ship. The provider order is a pluggable preference.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from .loader import Problem, Opp

K_RES = 3
# self-defense reservation scope: False = defended ships only (the paper); True = every TRACK pool
D4_ALL_POOLS = False


def d1_compatible(a: Opp, b: Opp) -> bool:
    return (b.tL >= a.tI + 3.0 or a.tL >= b.tI + 3.0 or
            (a.shooter != b.shooter and abs(a.tL - b.tL) <= 2.0))


class Decoder:
    def __init__(self, P: Problem, d4: bool = True):
        self.P = P
        self.d4 = d4 and bool((P.task.get("rule_switches") or {}).get("self_defense_channel_reservation"))
        self.load = {pool: [0] * len(v) for pool, v in P.cap.items()}
        self.others = defaultdict(lambda: None)   # pool -> list (TRACK use for non-own targets)
        self.used = defaultdict(int)
        self.count = defaultdict(int)
        self.sel: Dict[int, dict] = {}            # chain id -> spec
        self.by_target: Dict[int, List[int]] = defaultdict(list)
        self.pd_sel: Dict[int, None] = {}
        self.pd_count = defaultdict(int)
        self.aim_pid = [P.ship_ids[t.aim] for t in P.inst.threats]

    # ---------------------------------------------------------------- feasibility primitives
    def _fits(self, demands) -> bool:
        add = defaultdict(int)
        for pool, slots, amt, _ in demands:
            for s in slots:
                add[(pool, s)] += amt
        cap, load = self.P.cap, self.load
        for (pool, s), a in add.items():
            if load[pool][s] + a > cap[pool][s]:
                return False
        return True

    def _fits_d4(self, demands) -> bool:
        if not self.d4:
            return True
        add = defaultdict(int)
        for pool, slots, amt, other in demands:
            if other:
                for s in slots:
                    add[(pool, s)] += amt
        cap = self.P.cap
        for (pool, s), a in add.items():
            cur = self.others[pool]
            used = cur[s] if cur is not None else 0
            if used + a + K_RES > cap[pool][s]:
                return False
        return True

    def can_engage(self, e: int) -> bool:
        """cheap pre-checks: count, inventory, shoot-look-shoot."""
        op = self.P.opps[e]
        if e in self.sel or self.count[op.tgt] >= 2:
            return False
        if self.used[(op.shooter, op.weapon)] + 1 > self.P.inventory.get((op.shooter, op.weapon), 0):
            return False
        for f in self.by_target[op.tgt]:
            if not d1_compatible(self.P.opps[f], op):
                return False
        return True

    def probe(self, e: int, pref: Optional[Callable] = None) -> Optional[dict]:
        if not self.can_engage(e):
            return None
        P = self.P
        op = P.opps[e]
        demands = [(op.launch_pool, op.launch_slots, 1, False)]
        if not self._fits(demands):
            return None
        support = []
        aim = self.aim_pid[op.tgt]
        for sv in op.services:
            cands = sv.cands if pref is None else sorted(sv.cands, key=lambda c: pref(op, sv, c))
            picked = None
            for c in cands:
                if c.pool is None:
                    picked = (c, c.paths[0], []); break
                other = (sv.stype == "TRACK" and c.platform != aim and P.pool_type.get(c.pool) == "TRACK"
                         and (D4_ALL_POOLS or c.platform in P.ship_idx))   # constraint (6): defended ships only
                for path, relays in c.paths:
                    extra = [(c.pool, sv.slots, c.amount, other)] + [(r, sv.slots, 1, False) for r in relays]
                    if self._fits(demands + extra) and self._fits_d4(demands + extra):
                        picked = (c, (path, relays), extra); break
                if picked:
                    break
            if picked is None:
                return None
            demands.extend(picked[2])
            support.append((sv, picked[0], picked[1][0]))
        providers = frozenset(c.platform for _, c, _ in support)
        return {"demands": demands, "support": support, "providers": providers}

    def commit(self, e: int, spec: dict):
        op = self.P.opps[e]
        for pool, slots, amt, other in spec["demands"]:
            L = self.load[pool]
            for s in slots:
                L[s] += amt
            if other and self.d4:
                if self.others[pool] is None:
                    self.others[pool] = [0] * len(L)
                O = self.others[pool]
                for s in slots:
                    O[s] += amt
        self.used[(op.shooter, op.weapon)] += 1
        self.count[op.tgt] += 1
        self.by_target[op.tgt].append(e)
        self.sel[e] = spec

    def remove(self, e: int) -> dict:
        op = self.P.opps[e]
        spec = self.sel.pop(e)
        for pool, slots, amt, other in spec["demands"]:
            L = self.load[pool]
            for s in slots:
                L[s] -= amt
            if other and self.d4:
                O = self.others[pool]
                for s in slots:
                    O[s] -= amt
        self.used[(op.shooter, op.weapon)] -= 1
        self.count[op.tgt] -= 1
        self.by_target[op.tgt].remove(e)
        return spec

    # ---------------------------------------------------------------- point defence
    def can_pd(self, i: int) -> bool:
        P = self.P
        o = P.pd.get(i)
        if o is None or i in self.pd_sel:
            return False
        if self.pd_count[o.platform] + 1 > P.pd_ammo.get(o.platform, 0):
            return False
        return self._fits([(o.pool, o.slots, 1, False)]) if o.pool in P.cap else True

    def add_pd(self, i: int):
        o = self.P.pd[i]
        if o.pool in self.P.cap:
            for s in o.slots:
                self.load[o.pool][s] += 1
        self.pd_count[o.platform] += 1
        self.pd_sel[i] = None

    def remove_pd(self, i: int):
        o = self.P.pd[i]
        if o.pool in self.P.cap:
            for s in o.slots:
                self.load[o.pool][s] -= 1
        self.pd_count[o.platform] -= 1
        del self.pd_sel[i]

    # ---------------------------------------------------------------- export
    def solution(self, tag="PLAN") -> dict:
        P = self.P
        eng, sup, n = [], [], 0
        for e in sorted(self.sel):
            op = P.opps[e]; n += 1
            aid = "%s_EA_%05d" % (tag, n)
            eng.append({"action_id": aid, "target_id": P.target_ids[op.tgt], "shooter_platform_id": op.shooter,
                        "launcher_asset_id": op.launcher, "weapon_type_id": op.weapon, "salvo_id": op.salvo,
                        "quantity": 1, "launch_time_s": op.tL, "intercept_time_s": op.tI,
                        "source_opportunity_ref": op.oid})
            for sv, c, path in self.sel[e]["support"]:
                sup.append({"support_action_id": "%s_SA_%06d" % (tag, len(sup) + 1), "engagement_action_id": aid,
                            "service_template_id": sv.template, "service_type": sv.stype,
                            "provider_asset_id": c.asset, "provider_platform_id": c.platform,
                            "start_time_s": sv.start, "end_time_s": sv.end, "point_time_s": sv.point,
                            "communication_link_ids": list(path)})
        pda = [{"point_defense_action_id": "%s_PD_%05d" % (tag, k + 1), "target_id": P.target_ids[i],
                "defending_platform_id": P.pd[i].platform, "selected": True} for k, i in enumerate(sorted(self.pd_sel))]
        return {"solution_schema_version": "nad-solution-1.0", "instance_id": P.core["instance_id"],
                "task_id": P.task["task_id"], "engagement_actions": eng, "support_actions": sup,
                "handoff_actions": [], "communication_actions": [],
                "solver_metadata": {"solver_name": "NONE", "algorithm_name": "NONE", "algorithm_version": "NONE"},
                "optional_native_payload": {"point_defense_actions": pda, "objective_version": "O1-D3-KS-PD-v1"}}
