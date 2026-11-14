# Benchmarks

Timing harness for the SAM pipeline on synthetic data, used to verify that
code changes do not regress performance and to characterize scaling.

## Running

```bash
# Benchmark this package (the checkout containing benchmarks/):
python benchmarks/bench.py bench_reviewed.json

# Benchmark a different build, e.g. the original package:
python benchmarks/bench.py ../.. bench_original.json

# Compare two result files (per-stage timings + speedup ratios):
python benchmarks/compare_bench.py bench_original.json bench_reviewed.json
```

Run the two builds **sequentially**, not in parallel: concurrent runs contend
for CPU and add noise that swamps real differences.

## What is measured

- `sam_match` pipeline (four treatment groups, eight covariates): GPS fit,
  candidate search, matching, evaluation (SMD + AUC), cohort extraction +
  outcome analysis. Sizes: n = 5,000 / 20,000 / 50,000.
- `match_3way` (three groups): the KD-tree three-way matcher, which is the
  most expensive stage. Sizes: n = 20,000 / 50,000.

Each stage is timed over three repetitions; medians are reported. The
synthetic data uses a fixed seed, so matching rates are identical across
builds and any timing difference is attributable to the code.

## Findings (0.1.2 vs 0.2.0, Apple silicon, Python 3.13)

The two builds are performance-equivalent: every stage within ±4% (run-to-run
noise). The 0.2.0 changes are confined to the AUC diagnostic (a vectorized
clip on a small array) and add no measurable cost.

Scaling characteristics, shared by both builds:

| stage | n = 5k | n = 20k | n = 50k |
|---|--:|--:|--:|
| `sam_match` (heap greedy) | 0.09 s | 0.40 s | 1.13 s (near-linear) |
| `match_3way` (KD-tree, Python) | — | 2.2 s | 11.1 s (superlinear) |

The three-way matcher is the bottleneck at scale: its candidate loop is
Python-level. The `sam_match` path scales near-linearly and is the one to use
for large studies.
