"""Load an NNF-WPB scenario file and task file into the structures used by the engine and the
decoder: candidate chains with their support services, capacities, inventories, CIWS
opportunities, and ship values (slot-based occupancy, one provider per service window,
relay pools along the communication path)."""
from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .instance import Ship, Threat, Chain, Instance


def _read_json(path):
    """plain, xz-compressed (.xz), or gzip-compressed (.gz) JSON"""
    if str(path).endswith('.xz'):
        import lzma
        with lzma.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    if str(path).endswith('.gz'):
        import gzip
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    with open(path, encoding='utf-8') as f:
        return json.load(f)

EPSV = 1e-9


def relation_rows(core, relation_type):
    return [row for rel in core["capability_relations"] if rel["relation_type"] == relation_type
            for row in rel["table_entries"]]


@dataclass
class Cand:                 # one qualified provider option for one service of one opportunity
    asset: str
    platform: str
    pool: Optional[str]     # resource pool consumed (None if template has no resource demand)
    amount: int
    paths: List[Tuple[Tuple[str, ...], Tuple[str, ...]]]  # (link ids, relay pools) sorted by length


@dataclass
class Service:
    template: str
    stype: str
    start: Optional[float]
    end: Optional[float]
    point: Optional[float]
    slots: Tuple[int, ...]
    cands: List[Cand]


@dataclass
class Opp:
    oid: str
    tgt: int
    shooter: str
    weapon: str
    launcher: str
    salvo: str
    tL: float
    tI: float
    p: float
    mode: str
    launch_pool: Optional[str]
    launch_slots: Tuple[int, ...]
    services: List[Service]
    cost: float
    duty_violation: int = 0


@dataclass
class PDOpp:
    platform: str
    q: float
    pool: str
    slots: Tuple[int, ...]
    t0: float
    t1: float


@dataclass
class Problem:
    core: dict
    task: dict
    inst: Instance                 # engine instance (ships = defended platforms)
    opps: List[Opp]                # index == chain id in inst.chains
    ship_ids: List[str]            # ship index -> platform id
    ship_idx: Dict[str, int]
    target_ids: List[str]
    cap: Dict[str, List[int]]      # pool -> capacity per slot
    pool_type: Dict[str, str]
    inventory: Dict[Tuple[str, str], int]
    pd: Dict[int, PDOpp]           # threat index -> point-defence opportunity
    pd_ammo: Dict[str, int]
    E_inv: float


def _slot_ids(slots, start, end, point):
    if point is not None:
        for k, s in enumerate(slots):
            if s["start_s"] - 1e-9 <= point < s["end_s"] + 1e-9:
                return (k,)
        return ()
    return tuple(k for k, s in enumerate(slots) if s["start_s"] < end - 1e-9 and s["end_s"] > start + 1e-9)


def _anchor(o, name):
    return {"LAUNCH": o["launch_time_s"], "INTERCEPT": o["intercept_time_s"],
            "TERMINAL_START": o["terminal_start_s"]}[name]


def _allowed(o, service, platform):
    rules = o.get("mode_provider_constraints", {})
    allowed = rules.get(service)
    if allowed is None and service == "TRACK" and ("TRACK_AT_LAUNCH" in rules or "TRACK_AFTER_LAUNCH" in rules):
        allowed = sorted(set(rules.get("TRACK_AT_LAUNCH", []) + rules.get("TRACK_AFTER_LAUNCH", [])))
    return allowed is None or platform in allowed


def load(core_path: str, task_path: str, value_table: str = "MAIN", objective: str = "v11",
         max_opps_per_target: Optional[int] = None, o2_mode: str = "ammo",
         allowed_opps=None) -> Problem:
    """o2_mode: 'ammo'  O2 = interceptor expenditure only;
                'v11'   O2 = expenditure + omega_D x deviations from the zones of responsibility (eq. (4)),
                        judged at the intercept point with the self-defense exemption (used in the paper).
    max_opps_per_target / allowed_opps restrict the candidate chains (e.g. 12 per threat for the N12 study)."""
    core = _read_json(core_path)
    task = _read_json(task_path)
    slots = core["time_domain"]["standard_view"]["slots"]
    cap = {}
    for pool in core["resource_pools"]:
        vals = []
        for s in slots:
            act = [it["capacity"] for it in pool["capacity_profile"]
                   if it["start_s"] < s["end_s"] - 1e-9 and it["end_s"] > s["start_s"] + 1e-9]
            vals.append(int(min(act)) if act else 0)
        cap[pool["resource_pool_id"]] = vals
    pool_type = {p["resource_pool_id"]: p["resource_type_id"] for p in core["resource_pools"]}
    links = {l["link_id"]: l for l in core["communication_links"]}
    assets = {a["asset_id"]: a for a in core["assets"]}
    templates = {t["service_template_id"]: t for t in core["service_templates"]}
    effects = {r["opportunity_id"]: float(r["single_shot_probability"])
               for r in relation_rows(core, "WEAPON_TARGET_EFFECT_INPUT")}
    quals = defaultdict(list)
    for r in relation_rows(core, "PROVIDER_SERVICE_QUALIFICATION"):
        quals[(r["opportunity_id"], r["service_template_id"])].append(r)
    weapons = {w["weapon_type_id"]: w for w in core["weapon_types"]}
    # ships = defended platforms
    plats = [p for p in core["platforms"] if p.get("is_defended_asset")]
    values = task["platform_value_tables"][value_table]
    ship_ids = [p["platform_id"] for p in plats]
    ship_idx = {pid: k for k, pid in enumerate(ship_ids)}
    ships = [Ship(V=float(values.get(p["value_class"], 0.0)), K=int(p.get("hits_to_disable", 1))) for p in plats]
    target_ids = [t["target_id"] for t in core["targets"]]
    tidx = {tid: k for k, tid in enumerate(target_ids)}
    pd = {}
    pd_ammo = {}
    for inv in core["inventories"]:
        if inv["weapon_type_id"] == "W_POINT_DEFENSE":
            pd_ammo[inv["platform_id"]] = int(inv["initial_quantity"]) - int(inv["reserved_quantity"])
    for item in core.get("extensions", {}).get("scenario", {}).get("point_defense_opportunities", []):
        k = tidx[item["target_id"]]
        t0, t1 = float(item["valid_from_s"]), float(item["valid_to_s"])
        pd[k] = PDOpp(item["defending_platform_id"], float(item["intercept_probability"]),
                      item["resource_pool_id"], _slot_ids(slots, t0, t1, None), t0, t1)
    threats = [Threat(aim=ship_idx[t["intended_asset_ids"][0]], t=float(t["impact_time_s"]),
                      q=(pd[k].q if k in pd else 0.0)) for k, t in enumerate(core["targets"])]
    # zones of responsibility (eq. (4)): judged at the intercept point
    duty = task.get("defense_duty_policy", {"mode": "OFF"})
    opps: List[Opp] = []
    by_target = defaultdict(list)
    rows = relation_rows(core, "WEAPON_TARGET_ENGAGEMENT_WINDOW")
    for o in rows:
        by_target[o["target_id"]].append(o)
    selected_rows = []
    for tid, lst in by_target.items():
        lst.sort(key=lambda r: (-effects.get(r["opportunity_id"], 0.0), r["intercept_time_s"], r["opportunity_id"]))
        if allowed_opps is not None:          # explicit candidate domain
            lst = [r for r in lst if r["opportunity_id"] in allowed_opps]
        selected_rows.extend(lst if max_opps_per_target is None else lst[:max_opps_per_target])
    selected_rows.sort(key=lambda r: r["opportunity_id"])
    plat_by_id = {p["platform_id"]: p for p in core["platforms"]}
    traj = {t["trajectory_id"]: t for t in core.get("trajectories", [])}
    pos_xy = {}
    for p in core["platforms"]:
        t = traj.get(p.get("state_trajectory_id"))
        if t is not None:
            assert t.get("representation") == "STATIC", "the self-defense exemption assumes stationary ships"
            pos_xy[p["platform_id"]] = tuple(t["initial_position_m"])
    carrier_xy = next((pos_xy[p["platform_id"]] for p in core["platforms"]
                       if p.get("platform_role") == "CARRIER" and p["platform_id"] in pos_xy), (0.0, 0.0, 0.0))

    def _dist(a, b):        # 3-D distance, zero-padded
        k = max(len(a), len(b)); a = list(a) + [0.0] * (k - len(a)); b = list(b) + [0.0] * (k - len(b))
        return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5
    for o in selected_rows:
        if o["shooter_platform_id"] not in ship_idx:
            continue
        launcher = assets[o["launcher_asset_id"]]
        lpools = [p for p in launcher.get("resource_pool_ids", []) if p in cap]
        lslots = _slot_ids(slots, None, None, o["launch_time_s"])
        if not lpools or not lslots:
            continue
        services = []
        ok = True
        for tmpl_id in o["service_template_ids"]:
            tm = templates[tmpl_id]
            st = _anchor(o, tm["trigger_anchor"]) + tm["start_offset_s"]
            en = _anchor(o, tm["end_anchor"]) + tm["end_offset_s"]
            pt = None
            if tm["demand_mode"] == "POINT":
                pt = st + tm["point_offsets_s"][0]; st = en = None
            sl = _slot_ids(slots, st, en, pt)
            cands = []
            for q in quals[(o["opportunity_id"], tmpl_id)]:
                a = assets.get(q["provider_asset_id"])
                if a is None or not _allowed(o, tm["service_type"], a["host_platform_id"]):
                    continue
                lo = pt if pt is not None else st
                hi = pt if pt is not None else en
                if not (q["valid_from_s"] <= lo + 1e-9 and q["valid_to_s"] >= hi - 1e-9):
                    continue
                paths = q.get("communication_path_options", [])
                paths = [()] if not paths else sorted(map(tuple, paths), key=lambda x: (len(x), x))
                pp = []
                for path in paths:
                    relays = tuple(pid for lid in path for pid in links[lid].get("resource_pool_ids", [])
                                   if pid in pool_type and pool_type[pid] == "RELAY")
                    pp.append((path, relays))
                if not tm.get("resource_demands"):
                    cands.append(Cand(a["asset_id"], a["host_platform_id"], None, 0, pp[:1]))
                    continue
                req = tm["resource_demands"][0]
                pool = next((pid for pid in a.get("resource_pool_ids", [])
                             if pid in cap and pid.endswith("_" + req["resource_type_id"])), None)
                if pool is None or not sl:
                    continue
                cands.append(Cand(a["asset_id"], a["host_platform_id"], pool, int(req["amount"]), pp))
            if not cands:
                ok = False; break
            cands.sort(key=lambda c: c.asset)
            services.append(Service(tmpl_id, tm["service_type"], st, en, pt, sl, cands))
        if not ok:
            continue
        w = weapons[o["weapon_type_id"]]
        cost = float(w.get("consumption_equivalent", 1.0))
        dv = 0
        if duty.get("mode") in ("SOFT", "HARD"):
            sp = plat_by_id[o["shooter_platform_id"]]
            band = sp.get("duty_band_m") or {}
            ip = o.get("intercept_position_m")
            if band and ip is not None:
                r = _dist(ip, carrier_xy)
                lo, hi = band.get("min_from_carrier_m"), band.get("max_from_carrier_m")
                outside = (lo is not None and r <= lo) or (hi is not None and r > hi)
                near = _dist(ip, pos_xy[o["shooter_platform_id"]]) <= float(sp.get("self_defense_radius_m", 10000.0))
                dv = int(outside and not near)     # deviation counted in eq. (4)
        ammo = cost
        if o2_mode == "v11" and duty.get("mode") == "SOFT":
            cost = ammo + float(duty.get("soft_penalty_per_violation", 0.0)) * dv
        opps.append(Opp(o["opportunity_id"], tidx[o["target_id"]], o["shooter_platform_id"], o["weapon_type_id"],
                        o["launcher_asset_id"], o["salvo_id"], float(o["launch_time_s"]), float(o["intercept_time_s"]),
                        effects[o["opportunity_id"]], o.get("engagement_mode", "OWN"), lpools[0], lslots, services, cost, dv))
        opps[-1].ammo = ammo
    chains = [Chain(tgt=op.tgt, sh=ship_idx[op.shooter], p=op.p, tI=op.tI, cost=op.cost) for op in opps]
    inst = Instance(ships, threats, chains, h=float(task["lethality_policy"]["hit_probability"])).prepare()
    inst.objective = objective
    inst.o2_mode = o2_mode
    inventory = {(x["platform_id"], x["weapon_type_id"]): int(x["initial_quantity"]) - int(x["reserved_quantity"])
                 for x in core["inventories"] if x["weapon_type_id"] != "W_POINT_DEFENSE"}
    E_inv = sum(q * float(weapons[w].get("consumption_equivalent", 1.0)) for (p, w), q in inventory.items() if w in weapons)
    return Problem(core, task, inst, opps, ship_ids, ship_idx, target_ids, cap, pool_type, inventory, pd, pd_ammo, E_inv)


def carrier_floor(P):
    """absolute carrier survival floor kappa * S^max_CV from the task file, or None if the rule is off."""
    r = P.task.get("carrier_survival_rule", {}) or {}
    if not r.get("enabled") or r.get("smax_cv") is None:
        return None
    return float(r.get("kappa", 0.0)) * float(r["smax_cv"])
