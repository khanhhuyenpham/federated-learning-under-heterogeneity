# Client-update geometry results

This directory contains the saved outputs for Experiment 05 under strong Dirichlet-induced heterogeneity.

## Configuration

The primary condition uses:

- 5 clients with full participation;
- Dirichlet `alpha = 0.1`;
- 5 local epochs and 20 communication rounds;
- `mu in {0, 0.1, 1}`;
- seed 42.

See [`config.json`](config.json) for the saved configuration and [`../../notebooks/05_client_update_geometry.ipynb`](../../notebooks/05_client_update_geometry.ipynb) for the complete analysis.

## Data dictionary

All primary tables are in [`alpha_0p1/`](alpha_0p1/).

| File | Granularity | Main quantity |
|---|---|---|
| `performance_E5_T20_seed42.csv` | one row per `mu` and round | Test loss and accuracy |
| `magnitude_E5_T20_seed42.csv` | one row per `mu`, round, and client | `||Delta_k||_2` |
| `pairwise_E5_T20_seed42.csv` | one row per `mu`, round, and unique client pair | `cos(Delta_i, Delta_j)` |
| `aggregate_alignment_E5_T20_seed42.csv` | one row per `mu`, round, and client | Alignment with the weighted server update |
| `loo_alignment_E5_T20_seed42.csv` | one row per `mu`, round, and client | Alignment with the weighted update of all other clients |
| `client_label_distribution.csv` | one row per client | Label counts and client size |
| `client3_scaling_performance_E5_T20_seed42.csv` | one row per scaling factor and round | Exploratory client-weight intervention |

Here `Delta_k = w_k - w_t`, where `w_t` is the round-start global model and `w_k` is the locally trained client model.

## Interpretation warning

Ordinary aggregate alignment partly includes the client update being evaluated. Leave-one-out alignment removes that self-contribution and is therefore more suitable for asking whether a client agrees with its peers.

The scaling experiment is post-hoc and single-seed. It supports a follow-up hypothesis, but it does not establish that down-weighting a low-alignment client will generally improve federated learning.

## Figures

The `figures/` directory contains publication-style plots used by the main README and Notebook 05. They can be regenerated from the saved CSV files without rerunning federated training.
