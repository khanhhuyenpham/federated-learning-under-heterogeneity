# Result Artifacts

This directory contains compact CSVs, figures, and supporting artifacts from
the project's research progression. The final claims in the root README are
based on [`balanced_benchmark_v1/`](balanced_benchmark_v1/).

## Final evidence

| Directory | Producer | Role |
|---|---|---|
| [`balanced_benchmark_v1/`](balanced_benchmark_v1/) | Notebooks 09-10 | Final 63-run FedAvg, FedProx, and SCAFFOLD benchmark across IID, label skew, quantity skew, and rotation shift |

The final folder contains:

- `combined/`: eight analysis-ready CSVs exported from the completed benchmark
  archive;
- `analysis/tables/`: six compact derived tables;
- `analysis/figures/`: seven final PNG figures;
- `README.md`: result-level protocol, findings, limitations, and reproduction
  notes.

The complete raw archives are release assets rather than normal repository
files:

- `fedprox_mu_pilot_v1_complete.zip`
- `balanced_benchmark_v1_complete.zip`

## Earlier artifacts

These directories are retained to show how the project developed. They are not
the source of the final cross-heterogeneity claims.

| Directory | Producer | Role |
|---|---|---|
| [`iid_baseline/`](iid_baseline/) | Notebook 01 | FedAvg baseline under IID data |
| [`shard_noniid/`](shard_noniid/) | Notebook 02 | Pathological shard non-IID exploration |
| [`dirichlet_noniid/`](dirichlet_noniid/) | Notebook 03 | Exploratory Dirichlet label-skew runs |
| [`fedprox/`](fedprox/) | Notebook 04 | Exploratory FedProx coefficient sweep |
| [`client_update_geometry/`](client_update_geometry/) | Notebook 05 | Single-seed update-geometry mechanism study |
| [`multiseed_fedprox/`](multiseed_fedprox/) | Notebook 06 | Earlier FedAvg/FedProx robustness study |
| [`scaffold_under_heterogeneity/`](scaffold_under_heterogeneity/) | Notebook 07 | Earlier SCAFFOLD comparison under one label-skew setting |

## Evaluation notes

The final benchmark uses the official MNIST test split as the clean global
test set. It also reports local-test metrics on held-out examples from each
client's federated environment. Under rotation shift, these evaluate different
domains and should not be collapsed into a single score.

All final uncertainty summaries are descriptive mean ± sample standard
deviation across seeds 42, 43, and 44. They are not formal significance tests.
