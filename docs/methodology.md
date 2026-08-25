# Methodology Notes

Federated learning trains one global model from client-local data. When clients have different label distributions, each client can prefer a different update direction. This statistical heterogeneity creates client drift: local training steps move toward client-specific objectives before the server averages the resulting models.

FedAvg accepts those local updates as they are and aggregates them with sample-size weights. In this repository, sample-size weighting means each training example contributes equally to the pooled empirical objective.

FedProx changes the local objective by adding a proximal penalty around the round-start global model. This discourages large local deviations. Smaller updates can reduce instability, but update shrinkage is not the same as better optimization: an update can be small and still point in an unhelpful direction, or large and useful for the pooled objective.

SCAFFOLD addresses drift differently. During local training, it corrects the client gradient-scale direction using

```text
g_i + c - c_i
```

where `g_i` is the local stochastic gradient direction, `c` is the server control variate, and `c_i` is the persistent client control variate. The correction is intended to reduce client-specific bias in local updates.

This repository uses the Option II client-control update:

```text
c_i^+ = c_i - c + (x - y_i) / (K_i eta)
```

Here `x` is the round-start global model, `y_i` is the client model after local training, `K_i` is the number of local optimizer steps, and `eta` is the learning rate. The term `(x - y_i) / (K_i eta)` estimates an average gradient-scale direction over the local trajectory. The persistent `c_i` tracks a client's characteristic gradient direction across rounds.

In the first SCAFFOLD round, all server and client controls are initialized to zero. The correction `c - c_i` is therefore zero, so the first local SCAFFOLD update matches the corresponding FedAvg local update when initialization and minibatch order are identical.

The SCAFFOLD implementation here is full-participation and sample-size weighted. Server model aggregation and server-control updates both use client sample-size weights. This matches the pooled empirical objective, but it can give larger clients more influence. For that reason, the analysis reports macro-average accuracy, worst-client accuracy, and cross-client variability alongside weighted accuracy.

The trade-off is extra state and communication. Each client stores a control variate with the same shape as the trainable model parameters, and the server stores its own control variate. Under this repository's communication accounting, SCAFFOLD sends model/control information in both directions, which is approximately twice the full-vector communication per round compared with the FedAvg/FedProx accounting used here.
