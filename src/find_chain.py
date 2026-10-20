"""Lowest-error path of N qubits on a fake backend's coupling graph (sum of two-qubit gate errors), as used to pick
the ibm_torino chain (BLOCKGEO_CHAIN). Qubits with a readout error above the optional cap are excluded. Given a
comma-separated chain instead of N, it prints the two-qubit and readout errors of that chain.
Usage: python find_chain.py torino cz 10 [readout_cap]
       python find_chain.py sherbrooke ecr 33,39,40,41,53,60,59,58,71,77"""
import sys

import qemlib as L


def find_chain(name, twoq, N, cap=1.0):
    """Returns (path, sum of its two-qubit errors, pair errors, backend target)."""
    tgt = L.load_fake_backend(name).target
    ok = {q for (q,), p in tgt["measure"].items() if p is not None and p.error is not None and p.error <= cap}
    err = {}
    for (a, b), props in tgt[twoq].items():
        if a in ok and b in ok and props is not None and props.error is not None and props.error < 0.5:
            e = min(props.error, err.get((b, a), 1.0))
            err[(a, b)] = err[(b, a)] = e
    adj = {}
    for a, b in err:
        adj.setdefault(a, set()).add(b)
    best = [float("inf"), None]

    def dfs(path, cost):
        if cost >= best[0]:
            return
        if len(path) == N:
            best[:] = [cost, list(path)]
            return
        for nb in adj.get(path[-1], ()):
            if nb not in path:
                path.append(nb)
                dfs(path, cost + err[(path[-2], nb)])
                path.pop()

    for q in adj:
        dfs([q], 0.0)
    return best[1], best[0], err, tgt


def chain_errors(name, twoq, chain):
    """Two-qubit gate error of every coupler of a given chain, in the calibrated direction that the noise model of
    noise_layout.py uses, and readout error of every qubit."""
    tgt = L.load_fake_backend(name).target
    g = tgt[twoq]
    two = []
    for a, b in zip(chain, chain[1:]):
        if (a, b) not in g and (b, a) not in g:
            sys.exit(f"no {twoq} gate between qubits {a} and {b}")
        two.append(g[(a, b)].error if (a, b) in g else g[(b, a)].error)
    return two, [tgt["measure"][(q,)].error for q in chain]


def main():
    name, twoq = sys.argv[1], sys.argv[2]
    if "," in sys.argv[3]:
        path = [int(q) for q in sys.argv[3].split(",")]
        two, ro = chain_errors(name, twoq, path)
        cost = sum(two)
    else:
        N = int(sys.argv[3])
        cap = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        path, cost, err, tgt = find_chain(name, twoq, N, cap)
        two, ro = [err[(a, b)] for a, b in zip(path, path[1:])], [tgt["measure"][(q,)].error for q in path]
    print("chain", ",".join(map(str, path)))
    print("two-qubit errors", [round(e, 5) for e in two], "sum", round(cost, 5))
    print("readout errors", [round(r, 4) for r in ro])


if __name__ == "__main__":
    main()
