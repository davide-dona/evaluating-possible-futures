# Improper Suffix Metrics

Probabilistic suffix prediction is often evaluated by scoring each sampled suffix with a point metric
and averaging over the samples. This repository shows empirically that such an evaluation does not
reward predicting the true suffix distribution: on a simulated loan application process, a model that
always predicts the most probable suffix scores at least as well as an oracle that samples from the
true distribution.

- `src/improper_suffix_metrics/markov_process.py`: a Markov process over activities with lognormal
  times, its sampler, its most probable suffix, and random probabilities and times on its
  structure.
- `src/improper_suffix_metrics/evaluation.py`: the predictions of any forecaster on every prefix of
  every test case, and their averaged metrics (mean DLS, suffix length MAE, remaining time MAE).
- `figures/loan_process.drawio`: the BPMN diagram of the loan application process, exported to
  `figures/loan_process.svg` for the notebook.
- `experiment.ipynb`: the oracle, the mode model, and their comparison on one setting and on random
  settings.

## Usage

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace experiment.ipynb
```

To study another process, build a `MarkovProcess` like the loan one in the notebook and pass it, or
`randomize` of it, to the notebook's `compare`.
