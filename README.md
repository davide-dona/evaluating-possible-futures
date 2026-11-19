# Evaluating possible futures

This project tests how suffix prediction is scored. A forecaster sees the start of a running case and imagines several ways it could continue. The project scores those continuations against what actually happened, using metrics that average a point score over the continuations and metrics that judge them as a distribution. It shows that the averaged point metrics reward forecasters that always predict the single most typical future.

## From event logs to forecasts

An event log records many runs of a business process. Each run is a **case**, and each step of a case is an **event**: an **activity**, such as "Register request", that happened at a timestamp. The code represents a case as its activities in time order, a `tuple[str, ...]`.

Cutting a case after its first few events splits it in two. The **prefix** is the part already observed. The **suffix** is the rest of the case, the future the forecaster has to predict. One case of five events gives four prefixes, of lengths 1 to 4, each with a non-empty suffix.

A **forecaster** reads a prefix and returns several suffixes that could follow it. Each one is a **draw**: one possible future. A sampling forecaster returns different draws, while a greedy forecaster takes the most likely activity at every step and so returns the same suffix every time. The draws of one prefix form a **forecast**, and the forecast stage writes one forecast per scored test prefix. Inside a forecaster, **sampling** is any random choice made while producing a draw, such as picking the next activity or a dropout mask.

A **process** gives the probability of every possible next activity after a prefix, or of `<END>` when the case is over. Applying it step by step produces a suffix. The synthetic dataset is simulated from a known process, which serves as an oracle forecaster. A Markov chain fitted on the train cases serves as a baseline.

The neural forecaster, the U-ED-LSTM, works on integer **tokens** instead of activity strings. The tokenizer, fitted on the train cases, gives each train activity a token and reserves three special tokens ahead of them:

- `<PAD>` fills the shorter sequences of a batch up to the length of the longest. The model never predicts it.
- `<UNK>` stands for any activity the train cases do not hold. When scoring, every unknown activity of an observed suffix reads as `<UNK>`.
- `<EOS>` ends a suffix. The model predicts it when it expects the case to be over.

The stages run in this order, each a script in `scripts/`: `simulate` (synthetic data only), `preprocess`, `train`, `forecast`, `collapse`, `evaluate`, `tables`.

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
