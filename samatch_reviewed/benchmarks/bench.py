"""Benchmark one samatch build: full pipeline on synthetic data.

Usage:
    python benchmarks/bench.py [package_dir] out.json

`package_dir` defaults to the repository root (the parent of this directory),
so a plain run benchmarks the package in this checkout. Pass another path to
benchmark a different build, e.g. the original package:

    python benchmarks/bench.py ../.. bench_original.json
    python benchmarks/bench.py ..       bench_reviewed.json

The directory is inserted at the front of sys.path and `samatch` is imported
from it, so this exercises exactly that build (not any copy in site-packages).

Each stage is timed over three repetitions and the median is reported. The
synthetic data has eight normal covariates, confounded treatment assignment,
and a binary outcome with a small treatment effect.
"""

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd


def make_data(n, k_groups, seed):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({f"x{j}": rng.normal(0, 1, n) for j in range(8)})
    # Confounded treatment assignment: covariates drive group choice.
    lp1 = 0.6 * data["x0"] - 0.4 * data["x1"]
    lp2 = -0.5 * data["x0"] + 0.3 * data["x2"]
    lp3 = 0.4 * data["x1"] + 0.2 * data["x3"]
    logits = np.column_stack([np.zeros(n), lp1, lp2, lp3][:k_groups])
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    r = rng.random((n, k_groups))
    choice = (r < probs).argmax(axis=1)
    data["treatment"] = [chr(65 + g) for g in choice]
    p = 0.15 + 0.1 * data["x0"] - 0.08 * data["x1"]
    p = np.clip(p + 0.05 * (choice == 1) - 0.04 * (choice == 2), 0.01, 0.99)
    data["y"] = (rng.random(n) < p).astype(int)
    return data


def timeit(fn, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench_pipeline(sam, n, k_groups, seed):
    data = make_data(n, k_groups, seed)
    x_vars = [f"x{j}" for j in range(8)]
    anchor = "A"

    fit = sam.estimate_gps_multinom(data, X_vars=x_vars, treatment_var="treatment", anchor_level=anchor)
    t_gps = timeit(lambda: sam.estimate_gps_multinom(data, X_vars=x_vars, treatment_var="treatment", anchor_level=anchor))

    search = sam.gps_candidate_search(data, fit["gps"], treatment_var="treatment", anchor_level=anchor, top_m=10, gps_space="logit")
    t_search = timeit(lambda: sam.gps_candidate_search(data, fit["gps"], treatment_var="treatment", anchor_level=anchor, top_m=10, gps_space="logit"))

    matched = sam.sam_match(data, search, X_vars=x_vars, treatment_var="treatment")
    t_match = timeit(lambda: sam.sam_match(data, search, X_vars=x_vars, treatment_var="treatment"))

    t_eval = timeit(lambda: sam.sam_evaluate(data, search, matched, fit["gps"], X_vars=x_vars, treatment_var="treatment"))

    def extract_and_effects():
        c = sam.extract_matched_data(data, search, matched, treatment_var="treatment", anchor_level=anchor)
        sam.sam_estimate_effects(c, outcome_var="y", treatment_var="treatment")

    t_cohort = timeit(extract_and_effects)

    return {
        "algorithm": "sam_match", "n": n, "k_groups": k_groups,
        "matching_rate": matched["matching_rate"],
        "max_possible_rate": matched["max_possible_rate"],
        "n_matched_sets": len(matched["matched"]),
        "t_gps": t_gps, "t_search": t_search, "t_match": t_match,
        "t_eval": t_eval, "t_cohort_effects": t_cohort,
    }


def bench_3way(sam, n, seed):
    data = make_data(n, 3, seed)
    x_vars = [f"x{j}" for j in range(8)]
    anchor = "A"

    fit = sam.estimate_gps_multinom(data, X_vars=x_vars, treatment_var="treatment", anchor_level=anchor)
    search = sam.gps_candidate_search(data, fit["gps"], treatment_var="treatment", anchor_level=anchor, top_m=10, gps_space="logit")

    def run():
        return sam.match_3way(data, search, fit["gps"], treatment_var="treatment", caliper="auto", gps_space="raw", top_n=10)

    times = []
    result = None
    for _ in range(3):
        t0 = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - t0)

    return {
        "algorithm": "match_3way", "n": n, "k_groups": 3,
        "matching_rate": result["matching_rate"],
        "n_matched_sets": len(result["matched"]),
        "t_match_3way": float(np.median(times)),
    }


def main():
    default_dir = pathlib.Path(__file__).resolve().parent.parent

    if len(sys.argv) > 1:
        package_dir = sys.argv[1].rstrip("/")
    else:
        package_dir = str(default_dir)

    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    else:
        out_path = str(pathlib.Path(__file__).resolve().parent / "bench_results.json")

    sys.path.insert(0, package_dir)
    import samatch as sam

    assert sam.__file__.startswith(package_dir), f"imported wrong build: {sam.__file__}"
    print(f"bench: {package_dir} (version {sam.__version__}, {sam.__file__})", file=sys.stderr)

    results = []
    for n in (5_000, 20_000, 50_000):
        r = bench_pipeline(sam, n, 4, seed=7)
        results.append(r)
        print(f"  pipeline n={n}: rate={r['matching_rate']:.3f} gps={r['t_gps']:.3f}s "
              f"search={r['t_search']:.3f}s match={r['t_match']:.3f}s eval={r['t_eval']:.3f}s "
              f"cohort+effects={r['t_cohort_effects']:.3f}s", file=sys.stderr)

    for n in (20_000, 50_000):
        r = bench_3way(sam, n, seed=11)
        results.append(r)
        print(f"  3way n={n}: rate={r['matching_rate']:.3f} match_3way={r['t_match_3way']:.3f}s", file=sys.stderr)

    with open(out_path, "w") as f:
        json.dump({"package_dir": package_dir, "version": sam.__version__, "results": results}, f, indent=2)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
