# Notebook Guide

Read the notebooks as a research progression. Notebooks 01-07 document
exploratory development and mechanistic analysis. Notebooks 08-10 are the final
validation, execution, and analysis workflow for the balanced benchmark.

| # | Notebook | Role | Default execution |
|---:|---|---|---|
| 01 | [`01_fedavg_from_scratch.ipynb`](01_fedavg_from_scratch.ipynb) | FedAvg foundation under IID data | Full sweep disabled by default |
| 02 | [`02_fedavg_shard_noniid.ipynb`](02_fedavg_shard_noniid.ipynb) | Pathological shard non-IID exploration | Missing runs are opt-in |
| 03 | [`03_fedavg_dirichlet_noniid.ipynb`](03_fedavg_dirichlet_noniid.ipynb) | Dirichlet label-skew exploration | Full sweep disabled by default |
| 04 | [`04_fedprox_under_heterogeneity.ipynb`](04_fedprox_under_heterogeneity.ipynb) | Initial FedProx coefficient sweep | Full sweep disabled by default |
| 05 | [`05_client_update_geometry.ipynb`](05_client_update_geometry.ipynb) | Single-seed update-geometry mechanism study | Loads saved artifacts unless explicitly rerun |
| 06 | [`06_multiseed_fedprox_robustness.ipynb`](06_multiseed_fedprox_robustness.ipynb) | Earlier FedAvg/FedProx robustness study | Full multi-seed run guarded |
| 07 | [`07_scaffold_under_heterogeneity.ipynb`](07_scaffold_under_heterogeneity.ipynb) | Earlier SCAFFOLD label-skew comparison | Full multi-seed run guarded |
| 08 | [`08_fedprox_mu_selection.ipynb`](08_fedprox_mu_selection.ipynb) | Final validation-only FedProx `mu` selection | Reads the pilot archive and does not retrain |
| 09 | [`09_balanced_heterogeneity_benchmark.ipynb`](09_balanced_heterogeneity_benchmark.ipynb) | Final fixed 63-run benchmark execution and archive creation | Designed for Kaggle GPU; do not rerun casually |
| 10 | [`10_balanced_benchmark_analysis.ipynb`](10_balanced_benchmark_analysis.ipynb) | Final cross-seed analysis | Reads combined CSVs/archive and does not train |

## Final workflow

1. Notebook 08 validates the FedProx pilot and locks `mu = 0.01`.
2. Notebook 09 restores or executes the fixed benchmark: seven conditions,
   three seeds, and three algorithms.
3. Notebook 10 rebuilds the final tables and figures from the combined CSVs.

Notebook 09 is the only final notebook intended for expensive GPU execution.
The compact final outputs are committed under
[`../results/balanced_benchmark_v1/`](../results/balanced_benchmark_v1/), so
normal inspection and documentation review do not require rerunning it.
