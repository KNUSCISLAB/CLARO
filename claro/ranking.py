"""Alarm-time channel ranking with the attribution score.

rc(i) = rho_i - 0.5 * mean parent rho + 0.5 * log(1 + |descendants within distance 4|)

where rho is the current 45-s residual vector and both graph terms use the
LLM-pruned graph.
"""
import numpy as np

# Channel-name prefixes of the HAI candidate set: P1/P2/P3 process channels and the
# boiler-internal DCS outputs (x1001/x1002/x1003). P4 (HIL simulator) is excluded.
CANDIDATE_PREFIXES = ('P1_', 'P2_', 'P3_', 'x1001_', 'x1002_', 'x1003_')


def candidate_indices(names, evidence):
    """Indices of the rankable candidate channels (independent of attack labels)."""
    verified = {r['channel'] for r in evidence['verified']}
    return [i for i, n in enumerate(names) if n in verified or n.startswith(CANDIDATE_PREFIXES)]


def descendants(adj, idx, max_depth=4):
    """Channels reachable from ``idx`` by breadth-first search within ``max_depth`` steps.

    Each reachable channel counts once and ``idx`` itself is excluded, also when a
    cycle returns to it.
    """
    visited = set()
    queue = [idx]
    depth = {idx: 0}
    while queue:
        cur = queue.pop(0)
        for nb in np.where(adj[cur] > 0)[0]:
            if nb not in visited and depth.get(cur, 0) < max_depth:
                visited.add(nb)
                depth[nb] = depth[cur] + 1
                queue.append(nb)
    visited.discard(idx)
    return list(visited)


class CausalRanking:
    """Attribution score over a fixed candidate set."""

    def __init__(self, adj, eligible):
        degrees = adj.sum(0)
        # Column i holds 1/|Pa(i)| for each parent, so values @ weights is the parent mean.
        self.parent_weights = np.divide(adj, degrees, out=np.zeros_like(adj, dtype=float), where=degrees > 0)
        self.bonus = np.array([.5 * np.log1p(len(descendants(adj, i))) for i in range(adj.shape[0])])
        self.eligible = np.asarray(eligible, dtype=int)

    def components(self, values):
        """Residual, parent penalty, reachability bonus and total score."""
        values = np.asarray(values, dtype=float)
        penalty = -.5 * (values @ self.parent_weights)
        return values, penalty, self.bonus, values + penalty + self.bonus

    def rank(self, values, top=5):
        """Top candidates by total score (ties keep channel order)."""
        total = self.components(values)[-1]
        return self.eligible[np.argsort(-total[self.eligible], kind='stable')][:top].tolist()


def rank_by_scores(scores, eligible, top=5):
    """Top candidates by a plain score vector (ties keep channel order)."""
    eligible = np.asarray(eligible)
    return eligible[np.argsort(-scores[eligible], kind='stable')][:top].tolist()
