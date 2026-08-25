# Result Artifacts

This directory contains the committed configurations, raw metrics, derived
summaries, and figures produced by the repository's experiments. The final
claims in the [root README](../README.md) are based on the canonical artifacts
from Notebooks 05–07.

## Artifact map

| Directory | Producer | Role |
|---|---|---|
| [`client_update_geometry/`](client_update_geometry/) | Notebook 05 | Single-seed FedAvg/FedProx mechanism study under Dirichlet `alpha = 0.1` |
| [`multiseed_fedprox/`](multiseed_fedprox/) | Notebook 06 | Three-seed FedAvg/FedProx robustness study for `mu in {0.0, 0.1, 1.0}` |
| [`scaffold_under_heterogeneity/`](scaffold_under_heterogeneity/) | Notebook 07 | Three-seed comparison with the sample-size-weighted SCAFFOLD variant |

The earlier `iid_baseline/`, `shard_noniid/`, `dirichlet_noniid/`, and
`fedprox/` directories document the project's educational and exploratory
development. They are not the evidence used for the final multi-seed claims.

## Evaluation protocol

For Notebooks 05–07, each client's assigned examples from the MNIST training
collection are divided into local training, validation, and held-out test
subsets. The official MNIST test split is not used in the final client-level
comparison.

Notebook 05 uses seed 42. Notebooks 06 and 07 use matched seeds 42–44. Within
each of these seeded federated environments, method conditions reuse the same
client partition, client splits, initial model state, and seeded minibatch
schedule. One complete seeded environment—not an individual client or
communication round—is the experimental unit.

## Main CSV files

| File | Scope | Contents and use |
|---|---|---|
| `performance_history.csv` | Notebooks 05–07 | Round-level validation summaries, including weighted, macro, worst-client, and cross-client-dispersion metrics |
| `final_client_test_metrics.csv` | Notebooks 06–07 | Final held-out client-local test counts and accuracies; primary source for the root README result table |
| `client_performance_history.csv` | Notebook 05 only | Per-client validation trajectories for the single-seed mechanism study |
| `client_update_diagnostics.csv` | Notebooks 05–07 | Client-round update magnitude, relative magnitude, aggregation weight, aggregate alignment, and leave-one-out alignment |
| `pairwise_update_diagnostics.csv` | Notebooks 05–07 | Cosine similarity between each pair of client updates within a round |
| `round_diagnostics.csv` | Notebooks 05–07 | Round-level model, aggregate-update, participation, and method-specific summaries |
| `cost_summary.csv` | Notebook 07 | Modeled SCAFFOLD communication and persistent control-state costs |
| `partition_audit.csv` | Notebook 06 | Per-seed client sizes and label distributions used to inspect heterogeneity |

The method identifier is `mu` in the FedAvg/FedProx artifacts and `method` in
the SCAFFOLD artifacts. Some SCAFFOLD diagnostic files also include
control-variate magnitudes and local-step counts that do not apply to the
baselines.

## Derived summaries

| File | Purpose |
|---|---|
| `client_update_geometry/alpha_0p1/seed_42/comparison_summary.csv` | Compact single-seed comparison from Notebook 05 |
| `multiseed_fedprox/seed_mu_summary.csv` | One row per seed and FedAvg/FedProx coefficient |
| `multiseed_fedprox/across_seed_summary.csv` | Mean and sample standard deviation across seeds for each coefficient |
| `multiseed_fedprox/paired_mu_0p1_minus_0p0.csv` | Paired moderate-FedProx minus FedAvg differences within each seed |
| `scaffold_under_heterogeneity/seed_method_summary.csv` | One row per seed and method for the final comparison |
| `scaffold_under_heterogeneity/across_seed_summary.csv` | Mean and sample standard deviation across seeds for each method |
| `scaffold_under_heterogeneity/paired_scaffold_minus_fedavg.csv` | Paired SCAFFOLD minus FedAvg differences within each seed |

These files are derived for convenient inspection. The root README test table
can be independently recomputed from the two canonical
`final_client_test_metrics.csv` files.

For each seed and method:

- weighted accuracy is `sum(correct) / sum(total)` across clients;
- macro accuracy is the unweighted mean of the five client accuracies;
- worst-client accuracy is the minimum client accuracy;
- within-seed client dispersion uses the population standard deviation
  (`ddof = 0`).

The reported `mean ± SD` values then use the sample standard deviation across
the three seeds (`ddof = 1`).

## Seeded-world artifacts

Each `multiseed_fedprox/seed_*/` directory stores the inputs needed to preserve
the matched comparison:

- `partition_indices.json`;
- `train_indices.json`;
- `validation_indices.json`;
- `test_indices.json`;
- `client_label_counts.json`;
- `initial_model.pt`.

Notebook 07 reuses these saved partitions, splits, and initial model states
rather than generating new federated environments for SCAFFOLD.

## Figures

The root README uses
[`scaffold_under_heterogeneity/figures/validation_trajectories.png`](scaffold_under_heterogeneity/figures/validation_trajectories.png)
and
[`scaffold_under_heterogeneity/figures/geometry_comparison.png`](scaffold_under_heterogeneity/figures/geometry_comparison.png).

Notebook 05's single-seed mechanism figures are stored under
`client_update_geometry/alpha_0p1/seed_42/figures/`. The remaining figures in
the Notebook 06 and 07 directories support their paired performance, fairness,
and geometry analyses.

## Known exclusions and accounting caveats

Notebook 06's original archived per-client validation-history artifact is
excluded because an adapter mistakenly duplicated the aggregate performance
history. It is not part of the canonical artifact set, and no result in the
root README depends on it. The valid Notebook 06 evidence consists of the
final client-test table, aggregate validation history, partition audit, and
update-geometry diagnostics.

Runtime values should not be compared directly across FedAvg/FedProx and
SCAFFOLD. The baseline timer covers the complete experiment call, while the
SCAFFOLD timer excludes some evaluation and diagnostic construction.

Communication values use full-vector accounting for model and control-variate
transmission. They exclude protocol overhead, compression, secure aggregation,
and network effects, so they are estimates rather than measured traffic.

## Configuration scope

The canonical artifacts are tied to the documented configuration:

- five fully participating clients;
- Dirichlet `alpha = 0.1` statistical heterogeneity;
- 20 communication rounds;
- 5 local epochs per round;
- batch size 64;
- SGD with learning rate 0.01;
- an MLP with architecture `784 -> 128 -> 10`;
- seeds 42, 43, and 44 for the multi-seed studies;
- sample-size-weighted model aggregation for all methods;
- sample-size-weighted control aggregation for the SCAFFOLD variant.

New results should be compared with these artifacts only when differences in
the experimental controls and evaluation protocol are made explicit.

