"""Compact instance used by the evaluation engine.

Holds ships, threats, and candidate engagement chains. Support scheduling (configurations,
hand-overs, capacities) is handled by the decoder; each selected chain carries the set of
external provider ships chosen by the decoder.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from bisect import bisect_left
from typing import List, Tuple


@dataclass
class Ship:
    V: float          # value V_s
    K: int            # hits to disable K_s


@dataclass
class Threat:
    aim: int          # a(i): index of aimed ship
    t: float          # impact time t^imp_i
    q: float = 0.0    # point-defence kill probability q_i (0 if no PD opportunity)


@dataclass
class Chain:
    tgt: int          # i(e)
    sh: int           # s(e): shooter ship index
    p: float          # single-shot kill probability p_e
    tI: float         # intercept time t^I_e  (< t^imp of its target)
    cost: float = 1.0 # n_e * c_w(e)  (O2 ammunition part)


@dataclass
class Instance:
    ships: List[Ship]
    threats: List[Threat]
    chains: List[Chain]
    h: float
    # ---- derived (filled by prepare) ----
    order: List[int] = field(default_factory=list)          # threats by (t, id): global order
    rank: List[int] = field(default_factory=list)           # rank[i] = position of threat i in global order
    ship_thr: List[List[int]] = field(default_factory=list) # threats aimed at s, in global order
    pos: List[int] = field(default_factory=list)            # pos[i] = 1-based position of i in ship_thr[aim]
    ship_times: List[List[float]] = field(default_factory=list)
    chains_of: List[List[int]] = field(default_factory=list) # candidate chains per threat

    def prepare(self) -> "Instance":
        n = len(self.threats)
        self.order = sorted(range(n), key=lambda i: (self.threats[i].t, i))
        self.rank = [0] * n
        for r, i in enumerate(self.order):
            self.rank[i] = r
        S = len(self.ships)
        self.ship_thr = [[] for _ in range(S)]
        for i in self.order:
            self.ship_thr[self.threats[i].aim].append(i)
        self.pos = [0] * n
        for s in range(S):
            for k, i in enumerate(self.ship_thr[s]):
                self.pos[i] = k + 1
        self.ship_times = [[self.threats[i].t for i in self.ship_thr[s]] for s in range(S)]
        self.chains_of = [[] for _ in range(n)]
        self.sh_nu = [bisect_left(self.ship_times[c.sh], c.tI) for c in self.chains]
        for e, c in enumerate(self.chains):
            assert c.tI < self.threats[c.tgt].t, "intercept time must precede impact time (assumption A2)"
            self.chains_of[c.tgt].append(e)
        return self

    def nu(self, r: int, t: float) -> int:
        """number of threats aimed at r with impact time strictly before t."""
        return bisect_left(self.ship_times[r], t)

    def gprime(self, e: int, providers) -> Tuple[int, ...]:
        """G'_e: shooter + external provider ships, minus the aimed ship (Lemma S1 of the supplementary material)."""
        c = self.chains[e]
        a = self.threats[c.tgt].aim
        g = {c.sh, *providers}
        g.discard(a)
        return tuple(sorted(g))
