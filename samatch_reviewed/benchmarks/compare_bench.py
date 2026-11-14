"""Compare two bench result files: per-stage timings and speedup ratios.

Usage:
    python benchmarks/compare_bench.py bench_original.json bench_reviewed.json

Prints a per-stage timing table for both builds and the speedup ratio
(orig/new; >1 means the second build is faster). Ratios within a few percent
are run-to-run noise, not a real difference.
"""

import json
import sys

STAGES = ["t_gps", "t_search", "t_match", "t_eval", "t_cohort_effects", "t_match_3way"]


def key(r):
    return (r["algorithm"], r["n"])


def main():
    orig = json.load(open(sys.argv[1]))
    new = json.load(open(sys.argv[2]))

    o = {key(r): r for r in orig["results"]}
    nw = {key(r): r for r in new["results"]}

    all_keys = sorted(set(o) | set(nw), key=lambda k: (k[1], k[0]))

    print(f"{'build':<12} {'algo':<11} {'n':>7}  " + "  ".join(f"{s[2:]:>10}" for s in STAGES) + "   rate")
    for label, table in (("orig", o), ("new ", nw)):
        for k in all_keys:
            if k not in table:
                continue
            r = table[k]
            cells = "  ".join(
                f"{r[s]:>10.3f}" if s in r else f"{'-':>10}" for s in STAGES
            )
            rate = r.get("matching_rate", float("nan"))
            print(f"{label:<12} {k[0]:<11} {k[1]:>7}  {cells}   {rate:.3f}")

    print()
    print("speedup (orig/new per stage; >1 = new faster, <1 = new slower):")
    for k in all_keys:
        if k not in o or k not in nw:
            continue
        r, s = o[k], nw[k]
        parts = []
        for stage in STAGES:
            if stage in r and stage in s and s[stage] > 0:
                parts.append(f"{stage[2:]}={r[stage]/s[stage]:.3f}")
        if parts:
            print(f"  n={k[1]:>6} ({k[0]}): " + "  ".join(parts))


if __name__ == "__main__":
    main()
