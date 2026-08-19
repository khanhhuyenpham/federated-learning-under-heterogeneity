# Experiment guide

The notebooks are designed to be read in order. Together they move from implementation correctness to a mechanism-level question about heterogeneous federated optimization.

| # | Notebook | Main purpose | Key output |
|---:|---|---|---|
| 01 | [`01_fedavg_from_scratch.ipynb`](01_fedavg_from_scratch.ipynb) | Implement weighted FedAvg without an FL framework | IID local-epoch baseline |
| 02 | [`02_fedavg_shard_noniid.ipynb`](02_fedavg_shard_noniid.ipynb) | Introduce pathological shard non-IID data | IID versus severe label concentration |
| 03 | [`03_fedavg_dirichlet_noniid.ipynb`](03_fedavg_dirichlet_noniid.ipynb) | Control heterogeneity using Dirichlet `alpha` | Reusable `alpha = 0.1` partition |
| 04 | [`04_fedprox_under_heterogeneity.ipynb`](04_fedprox_under_heterogeneity.ipynb) | Implement and sweep FedProx | Communication-round convergence comparison |
| 05 | [`05_client_update_geometry.ipynb`](05_client_update_geometry.ipynb) | Measure update magnitude and directional alignment | Main mechanism-level analysis |

## Recommended reading paths

- **Project reviewer:** read Notebook 05, then Notebook 04.
- **Implementation learner:** start with Notebook 01 and proceed in order.
- **Results-only reader:** use the plots and CSVs under [`../results/`](../results/).

## Reproducibility note

The experiments reuse saved initial weights and client partitions where controlled comparison requires them. Expensive training cells are not intended to run automatically when a notebook is opened; saved results are included so the analysis remains inspectable.
