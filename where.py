#!/usr/bin/env python3
"""Ask Core ML which device each op actually runs on. No inference involved --
this is the compute planner's own assignment, not a timing inference."""
import collections

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan


def main(pkg="encoder.mlpackage"):
    for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_ONLY):
        m = ct.models.MLModel(pkg, compute_units=units)
        plan = MLComputePlan.load_from_path(path=m.get_compiled_model_path(),
                                            compute_units=units)
        main_fn = plan.model_structure.program.functions["main"]
        tally, byop = collections.Counter(), collections.defaultdict(collections.Counter)
        for op in main_fn.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            dev = type(usage.preferred_compute_device).__name__.replace("MLComputeDevice", "")
            tally[dev] += 1
            byop[op.operator_name][dev] += 1
        total = sum(tally.values())
        print(f"--- {str(units).split('.')[-1]} ---")
        for dev, n in tally.most_common():
            print(f"  {dev:14s} {n:4d} ops ({100*n/total:.1f}%)")
        strays = {k: dict(v) for k, v in byop.items() if "CPU" in str(v)}
        if strays and "NE" in str(units):
            print(f"  not on ANE: {strays}")


if __name__ == "__main__":
    main()
