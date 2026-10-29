# Improper Suffix Metrics

Probabilistic suffix prediction is often evaluated by scoring each sampled suffix with a point metric
and averaging over the samples. This repository shows empirically that such an evaluation does not
reward predicting the true suffix distribution: on a simulated loan application process, a model that
always predicts the most probable suffix scores at least as well as an oracle that samples from the
true distribution.

- `src/improper_suffix_metrics/markov_process.py`: a Markov process over activities with lognormal
  times, its sampler, and its most probable suffix.
- `src/improper_suffix_metrics/model.py`: process graphs, such as the loan application one, and the
  random Markov processes drawn on them.
- `src/improper_suffix_metrics/evaluation.py`: the oracle, the mode model, and the averaged metrics
  (mean DLS, suffix length MAE, remaining time MAE) over every prefix of every test case.
- `experiment.ipynb`: the experiment on one setting and on random settings.

## Usage

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace experiment.ipynb
```

To study another process, add its graph to `model.py` and pass it to `random_markov_process`, or
build a `MarkovProcess` with fixed probabilities and times, and pass the process to `compare`.
