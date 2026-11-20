# Evaluating Possible Futures

Experiments of the paper *Evaluating Possible Futures: Rethinking the Assessment of Probabilistic Suffix Prediction*.

**Probabilistic suffix prediction** is a novel paradigm within Predictive Process Monitoring 
(PPM). Rather than restricting the prediction to a single point estimate, it models a probability distribution over the possible future suffixes of an ongoing process instance, reflecting the uncertainty inherent in the process.

Current methods score sampled suffixes by their averaged Damerau-Levenshtein (DL) similarity to the ground-truth (GT) suffix. This can favour a degenerate predictor that collapses uncertainty over one that faithfully captures the process.

Our work adopts the energy score, a strictly proper scoring rule, instantiated with three distances:

- **exact match** gives a strictly proper score, but one blind to partial similarity between suffixes;
- **$n$-gram** gives a proper but not strictly proper score;
- **DL**, the standard choice in suffix prediction, gives an improper one.

The experiments benchmark these metrics on a synthetic log and four real event logs.

## Setup

Requires [Git LFS](https://git-lfs.com/) and [uv](https://docs.astral.sh/uv/) (Python 3.13).

```bash
git lfs install
git clone https://github.com/davide-dona/evaluating-possible-futures.git
cd evaluating-possible-futures
uv sync
```

## Data sources

The logs used come from 4TU.ResearchData and are stored as CSV in `data/<dataset>/original.csv`. 

| Dataset | Event log | DOI |
|---|---|---|
| `sepsis` | Sepsis Cases | [10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460](https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460) |
| `bpic12` | BPI Challenge 2012 | [10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f) |
| `bpic17` | BPI Challenge 2017 | [10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b) |
| `bpic19` | BPI Challenge 2019 | [10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1) |

## Reproducing the results

To reproduce the results, run the following commands in order. We will show the commands for Sepsis; for another log, change `dataset=sepsis` to `bpic12`, `bpic17`, `bpic19` or `synthetic`. Every script is configured with [Hydra](https://hydra.cc) from `config/`, which holds the settings and hyperparameters described in the paper.

### 1. Simulate the synthetic log

Apart from the real event logs listed in [Data sources](#data-sources), we add a synthetic log generated from a simple process with a long-term dependency. Since we know its true distribution, we can compare the predictors against it. To generate it, run:

```bash
uv run python scripts/simulate.py
```

### 2. Preprocess the logs

Before training, we clean each log and split its cases in time into a training and a test set. To preprocess a log, run:

```bash
uv run python scripts/preprocess.py dataset=sepsis
```

### 3. Train the U-ED-LSTM

On the real logs the true distribution is unknown, so we approximate it with a U-ED-LSTM trained on each log. Training needs a GPU and takes from a few minutes on Sepsis to about 16 hours on BPIC17. To train it, run:

```bash
uv run python scripts/train.py dataset=sepsis
```

### 4. Sample suffixes

Next, we let each predictor forecast the suffixes of every test prefix. On the real logs we compare the U-ED-LSTM (`sampled`) with a baseline that ignores the prefix (`unconditional`). On the synthetic log we also use the true process (`oracle_sampled`, `oracle_greedy`) and a first-order Markov chain (`markov_sampled`, `markov_greedy`). To run a predictor, pass its name to `forecaster`:

```bash
uv run python scripts/forecast.py dataset=sepsis forecaster=sampled
```

### 5. Collapse to the medoid

To see what happens when uncertainty collapses, we reduce the suffixes of the U-ED-LSTM to a single one, their medoid. This gives the `medoid` predictor. To compute it, run:

```bash
uv run python scripts/collapse.py dataset=sepsis
```

### 6. Score the predictions

We then score every predictor with the averaged DL similarity and the three energy scores. To score a predictor, pass its name to `forecasts`:

```bash
uv run python scripts/evaluate.py dataset=sepsis forecasts=sampled
```

### 7. Build the tables

Finally, once every log is scored, we collect the results into the tables of the paper, written to `outputs/tables/`. To build them, run:

```bash
uv run python scripts/tables.py
```

## Tensor dimensions

The shape annotations in the code, such as `Float[torch.Tensor, 'B T V']`, use one letter per dimension. Each letter has the same meaning everywhere:

| Symbol | Name | What it counts |
|---|---|---|
| `B` | batch size | Prefixes processed together. To sample `S` suffixes per prefix, the model copies every prefix `S` times and runs on `B * S` rows. |
| `P` | prefix length | Tokens in a prefix. In a batch, this is the longest prefix, and shorter ones are filled with `<PAD>`. |
| `L` | suffix length | Tokens or activities in a suffix, or in any single sequence. In a batch, this is the longest one. |
| `T` | steps | Positions a sequence model steps through. In training, these are the tokens the model predicts after each prefix. Inside an LSTM layer, they are the positions it reads: the `P` prefix tokens in the encoder, or a single token per decoder step. |
| `S` | draws per prefix | Possible futures a forecaster returns for one prefix. |
| `K` | logit samples | Logit vectors the model samples at every step to average its training loss. |
| `V` | vocabulary size | Distinct tokens: the train activities plus `<PAD>`, `<UNK>` and `<EOS>`. |
| `D` | embedding size | Numbers representing one token inside the network. |
| `I` | layer input size | Numbers one LSTM layer reads per step: `D` for the first layer, `H` for the layers above it. |
| `H` | hidden size | Numbers in an LSTM layer's memory, its hidden state and its cell state. |
| `W` | mask width | Features one dropout mask covers: `I` on a layer's input, `H` on its hidden state. |
| `4` | gates | The four gates of an LSTM layer, in the order input, forget, candidate, output. |
| `R` | kept rows | Rows kept from a batch, such as the suffixes still being sampled after the others reached `<EOS>`. |
| `N`, `M` | suffix counts | Sizes of the two groups of suffixes a distance compares, giving an `N` by `M` matrix. |
| `U` | distinct suffixes | Different suffixes among the draws of one prefix. |

A `*` before a name, as in `'*batch V'`, means any number of leading dimensions.
