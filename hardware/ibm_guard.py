"""Guarded access to IBM Quantum hardware: every job goes to the free Open Plan instance, and nothing is submitted
unless that instance is confirmed free. Also exposes the remaining free time.

    from ibm_guard import open_service, remaining_seconds
    service, backend = open_service("ibm_fez")
    remaining, limit, used = remaining_seconds(service)
"""
from qiskit_ibm_runtime import QiskitRuntimeService


def open_service(backend_name=None):
    """Service bound to the Open Plan instance; raises if that instance is not free or if a paid instance exists."""
    service = QiskitRuntimeService()
    insts = service.instances()
    free = [i for i in insts if i.get("plan") == "open" and i.get("pricing_type") == "free"]
    paid = [i for i in insts if i.get("pricing_type") != "free"]
    if paid:
        raise RuntimeError(f"paid instance present, refusing to run: {[i.get('name') for i in paid]}")
    if len(free) != 1:
        raise RuntimeError(f"expected exactly one free Open Plan instance, found {len(free)}")
    service = QiskitRuntimeService(instance=free[0]["crn"])
    backend = service.backend(backend_name) if backend_name else service.least_busy(operational=True,
                                                                                    simulator=False)
    return service, backend


def remaining_seconds(service):
    u = service.usage()
    return u["usage_remaining_seconds"], u["usage_limit_seconds"], u["usage_consumed_seconds"]
