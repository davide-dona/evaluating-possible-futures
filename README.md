# Improper Suffix Metrics

This repository tests whether mean Damerau-Levenshtein similarity (DLS), the metric commonly
used to score probabilistic suffix prediction, rewards a model for collapsing its predictive
distribution. It trains one U-ED-LSTM per event log and decodes every test prefix two ways from
the same checkpoint:

- **Sampled:** 100 suffixes per prefix, drawn with dropout and Gaussian logits active, as the
  U-ED-LSTM is meant to be used.
- **Greedy:** one suffix per prefix, with dropout off and the class of the highest logit mean
  taken at every step.

Both decodings are scored with mean DLS and with three fair energy scores. If DLS prefers greedy
decoding while the energy scores prefer sampling, then DLS rewards the collapsed forecast.

The four logs are Sepsis, BPIC12, BPIC17, and BPIC19, under `data/<dataset>/original.csv`.

## Install

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/), and
[Git LFS](https://git-lfs.com/) for the raw logs.

```sh
git lfs pull
uv sync --locked
```

## Run the pipeline

Each stage reads the files of the stages before it and writes its own under a fixed path, so a
rerun overwrites the previous one. After changing a stage's code or configuration, rerun every
stage from that one on. Run every command from the repository root.

```sh
uv run python -m pipelines.preprocess dataset=sepsis
uv run python -m pipelines.train dataset=sepsis
uv run python -m pipelines.generate dataset=sepsis
uv run python -m pipelines.evaluate dataset=sepsis
```

Hydra composes the YAML files under `config/` and accepts overrides such as `device=cpu`.
`--multirun dataset=sepsis,bpic12,bpic17,bpic19` runs a stage on every log. Each stage stores its
resolved configuration in `.hydra/config.yaml` inside its output directory.

| Stage | Writes | What it does |
| --- | --- | --- |
| preprocess | `data/<dataset>/processed/` | Drops the cases above the 98.5th percentile of length or the 95th percentile of duration, then the cases starting less than that duration before the end of the log. Splits the rest out of time and fits the activity vocabulary on train. |
| train | `outputs/<dataset>/train/` | Trains for 100 epochs and keeps the final weights, `model.pt`, with the losses of every epoch in `log.csv`. |
| generate | `outputs/<dataset>/generate/` | Writes `sampled.parquet` and `greedy.parquet`, one row per drawn suffix of every test prefix. |
| evaluate | `outputs/<dataset>/evaluate/` | Writes `scores.parquet`, one row per prefix and decoding, and prints the means. |

The out-of-time split puts every case still running at the start of the case at the 80th
percentile of case starts into test. A test case that started before that moment is scored only
on the prefixes that reach past it, since training may have seen its earlier events.

## Scores

Let $x_1, \dots, x_K$ denote the suffixes drawn for a prefix, $y$ the observed suffix, and $d$ a
distance between suffixes. Every score is computed per prefix and averaged over all test
prefixes, each weighed equally. The fair energy score charges the mean distance to the
observation, less half the mean distance between two different draws:

$$
\mathrm{ES}_d = \frac{1}{K} \sum_{i=1}^{K} d(x_i, y) - \frac{1}{2K(K-1)} \sum_{i \neq j} d(x_i, x_j),
$$

where lower is better. A single draw has no pairs, so greedy decoding is scored on the first term
alone. The three distances are:

| Distance | Between two suffixes |
| --- | --- |
| exact | 0 if they are equal, 1 otherwise |
| bigram | The multiset Jaccard distance between their bigrams, each suffix padded at both ends |
| dl | The Damerau-Levenshtein distance divided by the length of the longer suffix |

Mean DLS is one minus the first term of $\mathrm{ES}_{\mathrm{dl}}$, so both use the same distance
on the raw suffixes.
