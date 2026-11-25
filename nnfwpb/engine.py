"""Evaluation engine: the recursion (3) of the paper.

  alpha_e  = prod_{r in G'_e} S_r(t^I_e)
  ell_i    = sum_{Q subset X_i} (-1)^|Q| prod_{e in Q} p_e prod_{r in G'_Q} S_r(tau_r(Q))
  L_i      = ell_i * (1-q_i)^{y_i},  m_i = h L_i
  S_s(t)   = Phi_{K_s}(m_j : j aimed at s, t^imp_j < t)
  O1       = sum_s V_s S_s(t_end)

Provides
  * full evaluation of O1;
  * exact incremental evaluation on the influence cone of a change, with undo (Proposition 3);
  * marginal values lambda_i = dO1/dm_i by one reverse sweep (Proposition 3, eq. (11));
  * first-order estimates of the change of O1 using lambda.
"""
from __future__ import annotations
import heapq
from itertools import combinations
from typing import Dict, List, Tuple, Optional

from .instance import Instance

EPS = 1e-15


class Plan:
    """Fire-level decisions plus decoded external providers.
    X[i]   : tuple of selected chain ids on threat i (len <= 2)
    prov[e]: frozenset of external provider ships for selected chain e
    y[i]   : 0/1 point defence
    """
    def __init__(self, n_threats: int):
        self.X: List[Tuple[int, ...]] = [()] * n_threats
        self.prov: Dict[int, frozenset] = {}
        self.y: List[int] = [0] * n_threats

    def copy(self) -> "Plan":
        p = Plan(len(self.X))
        p.X = list(self.X); p.prov = dict(self.prov); p.y = list(self.y)
        return p


class Engine:
    def __init__(self, inst: Instance):
        self.I = inst
        S = len(inst.ships)
        self.K = [sh.K for sh in inst.ships]
        self.V = [sh.V for sh in inst.ships]
        self.plan: Optional[Plan] = None
        # state
        self.pi: List[List[Tuple[float, ...]]] = []
        self.Ssum: List[List[float]] = []
        self.ell: List[float] = []
        self.m: List[float] = []
        # readers[r][i] = max position nu of S_r read by target i's ell computation
        self.readers: List[Dict[int, int]] = [dict() for _ in range(S)]
        self.reads: List[List[Tuple[int, int]]] = []   # reads[i] = list of (r, nu) read by i
        self.O1 = 0.0
        self._log: list = []
        self.stats_recomputed = 0

    # ------------------------------------------------------------------ helpers
    def _terms(self, i: int, X: Tuple[int, ...], prov: Dict[int, frozenset]):
        """Inclusion-exclusion terms of ell_i (eq. (3b)).
        Returns list of (coef, [(r, nu), ...]) where term value = coef * prod S_r[nu]."""
        I = self.I
        compat = getattr(I, "objective", "v11") == "independent"
        if compat:   # reference variant used in Table S-I: aimed ship counted, two chains independent
            G = {e: tuple(sorted({I.chains[e].sh, *prov.get(e, frozenset())})) for e in X}
        else:
            G = {e: I.gprime(e, prov.get(e, frozenset())) for e in X}
        terms = []
        for k in range(len(X) + 1):
            for Q in combinations(X, k):
                coef = (-1.0) ** k
                if compat:
                    facs = []
                    for e in Q:
                        coef *= I.chains[e].p
                        facs.extend((r, I.nu(r, I.chains[e].tI)) for r in G[e])
                    terms.append((coef, facs))
                    continue
                tau: Dict[int, float] = {}
                for e in Q:
                    coef *= I.chains[e].p
                    for r in G[e]:
                        tI = I.chains[e].tI
                        if tI > tau.get(r, -1e300):
                            tau[r] = tI
                facs = [(r, I.nu(r, t)) for r, t in tau.items()]
                terms.append((coef, facs))
        return terms

    def _ell_from_terms(self, terms) -> float:
        v = 0.0
        for coef, facs in terms:
            t = coef
            for r, nu in facs:
                t *= self.Ssum[r][nu]
            v += t
        return v

    def _set_reads(self, i: int, terms):
        """update the reader index for target i (bookkeeping for incremental evaluation)."""
        for r, _ in self.reads[i]:
            self.readers[r].pop(i, None)
        mx: Dict[int, int] = {}
        for _, facs in terms:
            for r, nu in facs:
                if nu > mx.get(r, -1):
                    mx[r] = nu
        self.reads[i] = list(mx.items())
        for r, nu in mx.items():
            self.readers[r][i] = nu

    def _phi_step(self, vec: Tuple[float, ...], m: float, K: int) -> Tuple[float, ...]:
        out = [vec[0] * (1.0 - m)]
        for k in range(1, K):
            out.append(vec[k] * (1.0 - m) + vec[k - 1] * m)
        return tuple(out)

    # ------------------------------------------------------------------ full evaluation
    def evaluate(self, plan: Plan) -> float:
        I = self.I
        self.plan = plan.copy()
        S = len(I.ships)
        n = len(I.threats)
        self.pi = [[None] * (len(I.ship_thr[s]) + 1) for s in range(S)]
        self.Ssum = [[0.0] * (len(I.ship_thr[s]) + 1) for s in range(S)]
        for s in range(S):
            self.pi[s][0] = tuple([1.0] + [0.0] * (self.K[s] - 1))
            self.Ssum[s][0] = 1.0
        self.ell = [1.0] * n
        self.m = [0.0] * n
        self.readers = [dict() for _ in range(S)]
        self.reads = [[] for _ in range(n)]
        for i in I.order:
            terms = self._terms(i, self.plan.X[i], self.plan.prov)
            self._set_reads(i, terms)
            self.ell[i] = self._ell_from_terms(terms)
            q = I.threats[i].q if self.plan.y[i] else 0.0
            self.m[i] = I.h * self.ell[i] * (1.0 - q)
            s = I.threats[i].aim
            k = I.pos[i]
            self.pi[s][k] = self._phi_step(self.pi[s][k - 1], self.m[i], self.K[s])
            self.Ssum[s][k] = sum(self.pi[s][k])
        self.O1 = sum(self.V[s] * self.Ssum[s][-1] for s in range(S))
        return self.O1

    # ------------------------------------------------------------------ incremental evaluation
    def change(self, i: int, X: Tuple[int, ...], prov_new: Dict[int, frozenset], y: int) -> float:
        """Exact incremental re-evaluation after changing decisions on target i.
        prov_new gives providers for chains in X (others ignored). Returns new O1.
        All modifications are logged; call revert(token) to undo."""
        I, P = self.I, self.plan
        log = self._log
        log.append(('plan', i, P.X[i], {e: P.prov.get(e) for e in P.X[i]}, P.y[i], list(self.reads[i])))
        for e in P.X[i]:
            P.prov.pop(e, None)
        P.X[i] = tuple(X)
        for e in X:
            P.prov[e] = frozenset(prov_new.get(e, frozenset()))
        P.y[i] = y
        heap = [I.rank[i]]
        pending = {i}
        while heap:
            j = I.order[heapq.heappop(heap)]
            pending.discard(j)
            self.stats_recomputed += 1
            terms = self._terms(j, P.X[j], P.prov)
            if j == i:
                self._set_reads(j, terms)
            ell = self._ell_from_terms(terms)
            q = I.threats[j].q if P.y[j] else 0.0
            m = I.h * ell * (1.0 - q)
            log.append(('em', j, self.ell[j], self.m[j]))
            self.ell[j] = ell
            if abs(m - self.m[j]) <= EPS:
                self.m[j] = m
                continue
            self.m[j] = m
            s = I.threats[j].aim
            n0 = I.pos[j]
            K = self.K[s]
            old_end = self.Ssum[s][-1]
            ths = I.ship_thr[s]
            log.append(('pi', s, n0, self.pi[s][n0:], self.Ssum[s][n0:]))
            for k in range(n0, len(ths) + 1):
                self.pi[s][k] = self._phi_step(self.pi[s][k - 1], self.m[ths[k - 1]], K)
                self.Ssum[s][k] = sum(self.pi[s][k])
            self.O1 += self.V[s] * (self.Ssum[s][-1] - old_end)
            for tgt, numax in self.readers[s].items():
                if numax >= n0 and tgt not in pending:
                    pending.add(tgt)
                    heapq.heappush(heap, I.rank[tgt])
        return self.O1

    def token(self) -> Tuple[int, float]:
        return (len(self._log), self.O1)

    def revert(self, tok: Tuple[int, float]):
        L, O1 = tok
        P = self.P_ = self.plan
        while len(self._log) > L:
            rec = self._log.pop()
            if rec[0] == 'em':
                _, j, ell, m = rec
                self.ell[j] = ell; self.m[j] = m
            elif rec[0] == 'pi':
                _, s, n0, pis, ss = rec
                self.pi[s][n0:] = pis; self.Ssum[s][n0:] = ss
            else:
                _, i, X, prov, y, reads = rec
                for e in P.X[i]:
                    P.prov.pop(e, None)
                P.X[i] = X
                for e, pv in prov.items():
                    P.prov[e] = pv
                P.y[i] = y
                for r, _ in self.reads[i]:
                    self.readers[r].pop(i, None)
                self.reads[i] = reads
                for r, nu in reads:
                    self.readers[r][i] = nu
        self.O1 = O1

    def commit(self):
        self._log.clear()

    # ------------------------------------------------------------------ reverse-mode marginal values
    def marginal_values(self, indirect: bool = True) -> List[float]:
        """lambda_i = dO1/dm_i for every threat (Proposition 3), one backward pass.
        indirect=False keeps only the direct term (classical asset-based marginal value)."""
        I, P = self.I, self.plan
        S = len(I.ships)
        lam = [0.0] * len(I.threats)
        Sbar = [dict() for _ in range(S)]             # Sbar[r][nu] accumulated dO1/dS_{r,nu}
        run = [[self.V[s]] * self.K[s] for s in range(S)]  # pibar at current (last unprocessed) position
        for i in reversed(I.order):
            s = I.threats[i].aim
            n = I.pos[i]
            K = self.K[s]
            add = Sbar[s].pop(n, 0.0)
            pb = [v + add for v in run[s]]
            prev = self.pi[s][n - 1]
            mbar = 0.0
            for k in range(K):
                d = (prev[k - 1] if k >= 1 else 0.0) - prev[k]
                mbar += pb[k] * d
            lam[i] = mbar
            m = self.m[i]
            run[s] = [pb[k] * (1.0 - m) + (pb[k + 1] * m if k + 1 < K else 0.0) for k in range(K)]
            if indirect and P.X[i]:
                q = I.threats[i].q if P.y[i] else 0.0
                ellbar = mbar * I.h * (1.0 - q)
                for coef, facs in self._terms(i, P.X[i], P.prov):
                    if not facs:
                        continue
                    vals = [self.Ssum[r][nu] for r, nu in facs]
                    for a, (r, nu) in enumerate(facs):
                        prod_others = coef
                        for b, v in enumerate(vals):
                            if b != a:
                                prod_others *= v
                        Sbar[r][nu] = Sbar[r].get(nu, 0.0) + ellbar * prod_others
        # positions nu = 0 (constant) and leftovers need no propagation
        return lam

    # ------------------------------------------------------------------ first-order move estimate
    def ell_if(self, i: int, X: Tuple[int, ...], prov: Dict[int, frozenset]) -> float:
        """ell_i under hypothetical chain set X with current survival functions (exact locally)."""
        merged = dict(self.plan.prov); merged.update(prov)
        return self._ell_from_terms(self._terms(i, X, merged))

    def estimate(self, lam: List[float], i: int, X: Tuple[int, ...], prov: Dict[int, frozenset], y: int) -> float:
        """first-order estimate of Delta O1 for changing target i to (X, prov, y):
        exact local Delta m_i, linear propagation through lambda_i."""
        I = self.I
        q_new = I.threats[i].q if y else 0.0
        m_new = I.h * self.ell_if(i, X, prov) * (1.0 - q_new)
        return lam[i] * (m_new - self.m[i])
