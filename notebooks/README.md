# Notebook Guide

Read the notebooks as a research progression. Notebooks 01-04 document educational and exploratory development; Notebooks 05-07 provide the canonical evidence used in the root README.

Notebooks 01–04 are intentionally stored without execution outputs. Their code and explanatory markdown remain intact, while expensive experiments are opt-in. Notebooks 05–07 contain the regenerated saved-result analyses supporting the final conclusions.

|   # | Notebook                                                                         | Role                                                  | Default execution                                               |
| --: | -------------------------------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------- |
|  01 | [`01_fedavg_from_scratch.ipynb`](01_fedavg_from_scratch.ipynb)                   | Educational FedAvg implementation under IID data.     | Full sweep disabled by default.                                 |
|  02 | [`02_fedavg_shard_noniid.ipynb`](02_fedavg_shard_noniid.ipynb)                   | Educational shard non-IID experiment.                 | Missing runs are opt-in.                                        |
|  03 | [`03_fedavg_dirichlet_noniid.ipynb`](03_fedavg_dirichlet_noniid.ipynb)           | Exploratory Dirichlet heterogeneity sweep.            | Full sweep disabled by default.                                 |
|  04 | [`04_fedprox_under_heterogeneity.ipynb`](04_fedprox_under_heterogeneity.ipynb)   | Exploratory FedProx `mu` sweep.                       | Full sweep disabled by default.                                 |
|  05 | [`05_client_update_geometry.ipynb`](05_client_update_geometry.ipynb)             | Canonical single-seed client-update geometry study.   | Loads committed artifacts unless `RUN_FULL_EXPERIMENT=True`.    |
|  06 | [`06_multiseed_fedprox_robustness.ipynb`](06_multiseed_fedprox_robustness.ipynb) | Canonical multi-seed FedAvg/FedProx robustness study. | Full nine-condition run guarded by `RUN_FULL_MULTI_SEED=False`. |
|  07 | [`07_scaffold_under_heterogeneity.ipynb`](07_scaffold_under_heterogeneity.ipynb) | Canonical multi-seed comparison with SCAFFOLD.        | Full multi-seed run guarded by `RUN_MULTI_SEED=False`.          |

The final comparisons use client-local held-out subsets split from the MNIST training collection. The official MNIST test split is not used for Notebooks 05-07.

Notebook 06 excludes the invalid legacy per-client validation-history artifact produced by an earlier adapter mistake. Current claims use the final client-test table, aggregate validation history, and geometry diagnostics.
