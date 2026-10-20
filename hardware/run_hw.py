"""Hardware run on the free IBM Open Plan instance: uniform TFIM circuits of the simulated ensembles measured in the 9
covering settings on a chosen physical chain, plus two readout-calibration circuits.

Every (circuit, setting) pair is one Sampler PUB; pairs of training and test circuits are shuffled together so that
drift affects both alike, and submitted in jobs of at most 300 PUBs through ibm_guard (free instance only, budget
check before every job). Outcome frequencies of every job are written to <hardware dir>/<tag>/job<k>.npz as soon as
it finishes, so an interrupted run resumes without re-measuring. The ideal labels of the measured circuits are added
afterwards with make_labels.py <tag>.

usage: run_hw.py --backend ibm_fez --chain 129,118,109,110,111,98,91,92 --depth 1 --n-train 100 --n-test 30
                 --shots 1000
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler

import qemlib as L
from ibm_guard import open_service, remaining_seconds
from paths import HW as ROOT

MAX_PUBS, CAL_SHOTS, SEC_PER_SHOT = 300, 4000, 250e-6


def with_setting(qc, setting):
    """Append the basis change of one covering setting (X: H, Y: S^dagger then H) and measure all qubits."""
    out = qc.copy()
    for q, c in enumerate(setting):
        if c == "X":
            out.h(q)
        elif c == "Y":
            out.sdg(q)
            out.h(q)
    out.measure_all()
    return out


def calibration_snapshot(backend, chain):
    props = backend.properties()
    cq = [[props.t1(q) * 1e6, props.t2(q) * 1e6, props.gate_error("sx", q), props.readout_error(q)] for q in chain]
    ce = []
    for a, b in zip(chain, chain[1:]):
        e = [g for g in props.gates if g.gate == "cz" and set(g.qubits) == {a, b}]
        ce.append([[p.value for p in e[0].parameters if p.name == "gate_error"][0]])
    return dict(cq=cq, ce=ce, calibrated=str(props.last_update_date))


def counts_to_probs(counts, n):
    p = np.zeros(2 ** n)
    for k, v in counts.items():
        p[int(k.replace(" ", ""), 2)] = v
    return p / p.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    chain = [int(q) for q in args.chain.split(",")]
    n = len(chain)
    tag = args.tag or f"{args.backend}_d{args.depth}_tr{args.n_train}_te{args.n_test}_s{args.shots}"
    out_dir = os.path.join(ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    service, backend = open_service(args.backend)
    settings = L.measurement_settings(n, 2)
    train = L.make_circuits("tfim", n, args.depth, 200, "train")[:args.n_train]
    test = L.make_circuits("tfim", n, args.depth, 50, "test")[:args.n_test]
    circuits = [("train", i, qc) for i, qc in enumerate(train)] + [("test", i, qc) for i, qc in enumerate(test)]
    pairs = [(split, i, s) for split, i, _ in circuits for s in range(len(settings))]
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(pairs))
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        assert meta["order"] == order.tolist(), "existing run has a different layout"
    else:
        meta = dict(backend=args.backend, chain=chain, depth=args.depth, n_train=len(train), n_test=len(test),
                    shots=args.shots, cal_shots=CAL_SHOTS, settings=["".join(s) for s in settings],
                    pairs=pairs, order=order.tolist(), calibration=calibration_snapshot(backend, chain), jobs=[])
        json.dump(meta, open(meta_path, "w"), indent=1)

    # readout calibration circuits (all |0>, all |1>) go into the first job
    cal = [QuantumCircuit(n), QuantumCircuit(n)]
    cal[1].x(range(n))
    for c in cal:
        c.measure_all()
    tp = lambda qcs: transpile(qcs, backend=backend, initial_layout=chain, optimization_level=1, seed_transpiler=0)
    cal_isa = tp(cal)
    batches = [order[k:k + MAX_PUBS - 2] if k == 0 else order[k:k + MAX_PUBS] for k in
               [0] + list(range(MAX_PUBS - 2, len(order), MAX_PUBS))]
    print(f"{tag}: {len(pairs)} circuit-settings in {len(batches)} jobs; free time remaining "
          f"{remaining_seconds(service)[0]} s", flush=True)
    for k, idx in enumerate(batches):
        f = os.path.join(out_dir, f"job{k}.npz")
        if os.path.exists(f):
            print(f"job {k} already done", flush=True)
            continue
        qcs = []
        for j in idx:
            split, i, s = pairs[j]
            src = train[i] if split == "train" else test[i]
            qcs.append(with_setting(src, settings[s]))
        isa = tp(qcs)
        pubs = [(c, None, args.shots) for c in isa]
        if k == 0:
            pubs = [(c, None, CAL_SHOTS) for c in cal_isa] + pubs
        need = (len(idx) * args.shots + (2 * CAL_SHOTS if k == 0 else 0)) * SEC_PER_SHOT * 1.2 + 5
        rem = remaining_seconds(service)[0]
        if need > rem - 5:
            raise RuntimeError(f"job {k} needs about {need:.0f} s but only {rem} s remain")
        inst = [i for i in service.instances() if i["crn"] == service.active_instance()][0]
        assert inst["plan"] == "open" and inst["pricing_type"] == "free", "not the free instance"
        sampler = Sampler(mode=backend)
        sampler.options.twirling.enable_gates = True
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        t0 = time.time()
        job = sampler.run(pubs)
        print(f"job {k}: {len(pubs)} PUBs submitted as {job.job_id()} (needs about {need:.0f} s)", flush=True)
        res = job.result()
        data = [counts_to_probs(r.data.meas.get_counts(), n) for r in res]
        m = job.metrics()
        charge = m.get("usage", {}).get("qpu_charge_time_seconds")
        np.savez_compressed(f, idx=np.asarray(idx), probs=np.asarray(data[2:] if k == 0 else data, dtype=np.float32),
                            cal=np.asarray(data[:2], dtype=np.float32) if k == 0 else np.zeros(0))
        meta["jobs"].append(dict(k=k, job_id=job.job_id(), pubs=len(pubs), charge_s=charge,
                                 wall_s=time.time() - t0, timestamps=m.get("timestamps")))
        json.dump(meta, open(meta_path, "w"), indent=1, default=str)
        print(f"job {k} done: charged {charge} s, wall {time.time() - t0:.0f} s, free time remaining "
              f"{remaining_seconds(service)[0]} s", flush=True)
    print("RUN COMPLETE", tag, flush=True)


if __name__ == "__main__":
    main()
