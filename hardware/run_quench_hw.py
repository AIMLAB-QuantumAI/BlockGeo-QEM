"""Hardware measurement of the fixed-parameter quench circuit (theta_h = pi/4, theta_J = -pi/4) at the given Trotter
step counts, in the 9 covering settings, plus two readout-calibration circuits, in one job on the free instance.
Data: <hardware dir>/quench_<backend>/job0.npz and meta.json (same layout as run_hw.py, one "test" circuit per
depth). The training labels used by quench_hw.py are added afterwards with make_labels.py.

usage: run_quench_hw.py --backend ibm_fez --chain 129,118,109,110,111,98,91,92 --depths 1,2 --shots 1000
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
from run_hw import CAL_SHOTS, ROOT, SEC_PER_SHOT, calibration_snapshot, counts_to_probs, with_setting
from run_quench import TH_H, TH_J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--depths", default="1,2")
    ap.add_argument("--shots", type=int, default=1000)
    args = ap.parse_args()
    chain = [int(q) for q in args.chain.split(",")]
    n, depths = len(chain), [int(x) for x in args.depths.split(",")]
    out_dir = os.path.join(ROOT, f"quench_{args.backend}")
    os.makedirs(out_dir, exist_ok=True)
    service, backend = open_service(args.backend)
    settings = L.measurement_settings(n, 2)
    pairs = [(d, s) for d in depths for s in range(len(settings))]
    qcs = [with_setting(L.tfim_circuit(n, d, TH_H, TH_J), settings[s]) for d, s in pairs]
    cal = [QuantumCircuit(n), QuantumCircuit(n)]
    cal[1].x(range(n))
    for c in cal:
        c.measure_all()
    isa = transpile(cal + qcs, backend=backend, initial_layout=chain, optimization_level=1, seed_transpiler=0)
    pubs = [(c, None, CAL_SHOTS) for c in isa[:2]] + [(c, None, args.shots) for c in isa[2:]]
    need = (len(qcs) * args.shots + 2 * CAL_SHOTS) * SEC_PER_SHOT * 1.2 + 5
    rem = remaining_seconds(service)[0]
    print(f"{len(pubs)} PUBs, needs about {need:.0f} s, free time remaining {rem} s", flush=True)
    if need > rem - 5:
        raise RuntimeError("not enough free time")
    inst = [i for i in service.instances() if i["crn"] == service.active_instance()][0]
    assert inst["plan"] == "open" and inst["pricing_type"] == "free", "not the free instance"
    sampler = Sampler(mode=backend)
    sampler.options.twirling.enable_gates = True
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    t0 = time.time()
    job = sampler.run(pubs)
    print("submitted", job.job_id(), flush=True)
    res = job.result()
    data = [counts_to_probs(r.data.meas.get_counts(), n) for r in res]
    m = job.metrics()
    charge = m.get("usage", {}).get("qpu_charge_time_seconds")
    np.savez_compressed(os.path.join(out_dir, "job0.npz"), idx=np.arange(len(pairs)),
                        probs=np.asarray(data[2:], dtype=np.float32), cal=np.asarray(data[:2], dtype=np.float32))
    meta = dict(backend=args.backend, chain=chain, depths=depths, theta_h=TH_H, theta_J=TH_J, shots=args.shots,
                cal_shots=CAL_SHOTS, settings=["".join(s) for s in settings], pairs=pairs,
                calibration=calibration_snapshot(backend, chain),
                jobs=[dict(k=0, job_id=job.job_id(), pubs=len(pubs), charge_s=charge, wall_s=time.time() - t0,
                           timestamps=m.get("timestamps"))])
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1, default=str)
    print(f"done: charged {charge} s, wall {time.time() - t0:.0f} s, free time remaining "
          f"{remaining_seconds(service)[0]} s", flush=True)


if __name__ == "__main__":
    main()
