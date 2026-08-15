# IID baseline artifacts

This directory preserves the completed 5-client IID FedAvg baseline run.

## Exact run artifacts

- `config.json` — experiment configuration
- `summary.csv` — final metrics and rounds-to-target statistics
- `fedavg_iid_E1_history.csv` — exact per-round history for `E=1`
- `fedavg_iid_E5_history.csv` — exact per-round history for `E=5`
- `fedavg_iid_E10_history.csv` — exact per-round history for `E=10`
- `initial_model.pt` — initial global model state used for the controlled sweep
- `client_indices.pt` — exact IID partition; 5 clients × 12,000 MNIST training examples
- `fedavg_iid_E10_final.pt` — final global model checkpoint from the `E=10` run
- `accuracy_vs_round.png`, `accuracy_vs_local_epochs.png`, `loss_vs_round.png` — baseline figures

The three history CSVs and `.pt` files are the original artifacts exported from the completed Colab runtime, not reconstructed values.

This is a single-seed development baseline (`seed=42`), so it should not be presented as a statistically robust final benchmark.
