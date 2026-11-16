# Evaluating Possible Futures

This repository holds the experiments of the paper *Evaluating Possible Futures: Rethinking the
Assessment of Probabilistic Suffix Prediction*. A probabilistic suffix predictor forecasts how a
running case will continue as a set of sampled suffixes, since the same prefix can lead to
several plausible continuations. The paper scores such forecasts with three energy scores, each
built on a distance between suffixes. An energy score is a proper scoring rule: it rewards a
forecast for both the accuracy and the spread of its suffixes, so it cannot be improved by
collapsing on a single continuation. The usual metric, mean Damerau-Levenshtein similarity (DLS)
over the sampled suffixes, accounts for accuracy only.

To measure what this costs in practice, the repository trains one U-ED-LSTM per real event log
and scores three forecasts of every test prefix:

- **Sampled:** 100 suffixes per prefix, drawn from the U-ED-LSTM with dropout and Gaussian
  logits active, as the model is meant to be used.
- **Medoid:** one suffix per prefix, the sampled suffix with the highest mean DLS to all 100. It
  collapses the sampled forecast on a single continuation and uses nothing beyond its draws.
- **Unconditional:** 100 suffixes per prefix, drawn from the suffixes of the train split
  whatever the prefix.

Every forecast is scored with mean DLS and with the three energy scores. The medoid shows how
much mean DLS a forecast gains by suppressing alternative futures, and how the energy scores
charge it for that. The unconditional forecast spreads its draws while knowing nothing about the
running case, so it shows which energy scores tell a trained model from uninformed spread.

The four logs are Sepsis, BPIC12, BPIC17, and BPIC19, under `data/<dataset>/original.csv`.

## Install

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/), and
[Git LFS](https://git-lfs.com/) for the raw logs.

```sh
git lfs pull
uv sync --locked
```

## Run the pipeline

Every stage has its own script under `scripts/`, which reads the configuration and calls the
`possible_futures` package under `src/`. Each stage reads the files of the stages before it and writes its own under a fixed path, so a
rerun overwrites the previous one. After changing a stage's code or configuration, rerun every
stage from that one on. Run every command from the repository root.

```sh
uv run python scripts/preprocess.py dataset=sepsis
uv run python scripts/train.py dataset=sepsis
uv run python scripts/generate.py dataset=sepsis
uv run python scripts/collapse.py dataset=sepsis
uv run python scripts/baselines.py dataset=sepsis
uv run python scripts/evaluate.py dataset=sepsis
```

Hydra composes the YAML files under `config/` and accepts overrides such as `device=cpu`.
`--multirun dataset=sepsis,bpic12,bpic17,bpic19` runs a stage on every log. Each stage stores its
resolved configuration in `.hydra/config.yaml` inside its output directory.

| Stage | Writes | What it does |
| --- | --- | --- |
| preprocess | `data/<dataset>/processed/` | Drops the cases above the 98.5th percentile of length or the 95th percentile of duration, then every prefix ending less than that duration before the end of the log. Splits the rest out of time and fits the activity vocabulary on train. |
| train | `outputs/<dataset>/train/` | Trains for 100 epochs and keeps the final weights, `model.pt`, with the loss of every epoch in `log.csv`. |
| generate | `outputs/<dataset>/generate/` | Writes `sampled.parquet`, one row per drawn suffix of every test prefix. |
| collapse | `outputs/<dataset>/collapse/` | Writes `medoid.parquet`, the medoid of the sampled suffixes of every test prefix. |
| baselines | `outputs/<dataset>/baselines/` | Writes `unconditional.parquet`, the suffixes drawn from train for every test prefix. It needs no model. |
| evaluate | `outputs/<dataset>/evaluate/` | Writes `scores.parquet`, one row per prefix and decoding, and prints the means. It scores the prefixes in `workers` processes, 8 by default. |

`evaluate` scores every decoding in `outputs/<dataset>/generate/`, `outputs/<dataset>/collapse/`
and `outputs/<dataset>/baselines/`. A decoding is one Parquet file, named after it.

The medoid of a prefix is the sampled suffix with the lowest mean Damerau-Levenshtein distance,
divided by the length of the longer suffix, to all the sampled suffixes of that prefix. Mean DLS
is highest for a forecast that always returns the suffix with the highest expected DLS, and the
medoid is the estimate of that suffix from the draws. It is the decoding rule of BRIDGE (He et
al., *BRIDGE: A Bayes-Risk-Informed Process Decoder for Suffix Prediction*), with every sampled
suffix weighted by how often it was drawn.

The unconditional forecast draws uniformly from the prefix-suffix pairs of train, each prefix of
train counting once, as each test prefix does in a mean score.

The preprocessing follows Weytjens and De Weerdt, *Creating Unbiased Public Benchmark Datasets
with Data Leakage Prevention for Predictive Process Monitoring*. The end of a log is debiased
over a window as long as the longest kept case: a case starting in the window is dropped, and a
case running into it is scored only on the prefixes that end before it.

The out-of-time split puts every case still running at the start of the case at the 80th
percentile of case starts into test. A test case that started before that moment is scored only
on the prefixes that reach past it, since training may have seen its earlier events.

An activity of test that train does not hold is encoded as `<UNK>`, which is also an
output class of the model, and is scored as `<UNK>`. A forecast can therefore predict that an
activity unseen in train comes next, but not which one.

## Synthetic log

On a real log the true distribution of the suffixes is unknown. The synthetic log comes from a
known process, so forecasts drawn from the truth can be scored. No model is trained on it:

```
        ┌─ B ─┐            ┌─ E ─┐
  A ────┤     ├── D ───────┤     ├── G
        └─ C ─┘            └─ F ─┘
```

A case takes B with probability 0.7 and C otherwise. After D, it takes E with probability 0.9 if
it went through B and 0.3 if it went through C, so a first-order Markov chain, which sees only D,
cannot tell the two apart. The probabilities and the 20,000 cases are set in
`config/dataset/synthetic.yaml`.

```sh
uv run python scripts/simulate.py
uv run python scripts/preprocess.py dataset=synthetic
uv run python scripts/baselines.py dataset=synthetic
uv run python scripts/evaluate.py dataset=synthetic
```

`simulate` writes `data/synthetic/original.csv`, which the other stages read like any original
log. On a log whose configuration holds its process, `baselines` writes four more decodings:

| Decoding | Suffixes of every test prefix |
| --- | --- |
| oracle_sampled | 100 draws from the true process |
| oracle_greedy | 1 suffix, the most probable activity under the true process at every step |
| markov_sampled | 100 draws from a first-order Markov chain fitted on train |
| markov_greedy | As oracle_greedy, under the Markov chain |

## Tables

```sh
uv run python scripts/tables.py
```

`tables` writes the three tables of the paper to `outputs/tables/`:

| Table | Rows | Columns |
| --- | --- | --- |
| `real.tex` | The sampled, medoid and unconditional decodings, grouped by real log | Mean DLS and the three energy scores, the best mean of every log in bold |
| `synthetic.tex` | The sampled and greedy decodings of the true process and of the Markov chain on the synthetic log | As `real.tex` |
| `properties.tex` | Every preprocessed log | Cases, events, variants, activities, and case length and duration |

The tables need the `booktabs`, `multirow`, `dcolumn` and `siunitx` packages. The headers use
the metric macros of the paper's preamble, and the rows use these macros, which the preamble
must define: `\Sampled`, `\Medoid`, `\Unconditional`, `\OracleSampled`,
`\OracleGreedy`, `\MarkovSampled` and `\MarkovGreedy`. The logs and the decodings of every
table, their printed names, and the order of the rows are set in `config/tables.yaml`.

## Scores

Let $x_1, \dots, x_K$ denote the suffixes drawn for a prefix, $y$ the observed suffix, and $d$ a
distance between suffixes. Every score is computed per prefix and averaged over all test
prefixes, each weighed equally. The fair energy score charges the mean distance to the
observation, less half the mean distance between two different draws:

$$
\mathrm{ES}_d = \frac{1}{K} \sum_{i=1}^{K} d(x_i, y) - \frac{1}{2K(K-1)} \sum_{i \neq j} d(x_i, x_j),
$$

where lower is better. A single draw has no pairs, so a decoding of one suffix per prefix, such
as the medoid, is scored on the first term alone. The three distances are:

| Distance | Between two suffixes |
| --- | --- |
| exact | 0 if they are equal, 1 otherwise |
| bigram | Half the L1 distance between their relative bigram frequencies, the share each bigram takes of a suffix padded at both ends |
| dl | The Damerau-Levenshtein distance divided by the length of the longer suffix |

Mean DLS is one minus the first term of $\mathrm{ES}_{\mathrm{dl}}$, so both use the same distance
on the raw suffixes.
