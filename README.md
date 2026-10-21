# Character-Level Language Model from Scratch

A character-level language model built with **PyTorch** and trained on
Andrej Karpathy's **Tiny Shakespeare** dataset. I built this project to
understand the foundations of language modelling and how the core
components of a small decoder-style Transformer work together.

> **Project status:** Learning project. The model learns character-level
> patterns and generates text, but the output is not consistently
> coherent. The goal is to understand the architecture and training
> process, not to present this as a production-ready LLM.

## Why I Built This

I wanted to move beyond using language models as black-box APIs and
understand what happens inside one. Building a small model from scratch
gave me a practical way to explore tokenization, embeddings, attention,
training, and text generation while learning the fundamentals behind
GPT-style models.

## Goals

-   Understand how text is represented as tokens and training examples.
-   Implement the core components of a small decoder-style language
    model.
-   Train the model to predict the next character and track validation
    loss.
-   Experiment with context length and embedding dimensions.
-   Understand how model design and sampling affect generated text.

## How It Works

The notebook loads Tiny Shakespeare, maps each unique character to an
integer, and creates training examples from sliding context windows. The
model combines token and positional embeddings, causal self-attention,
and a feed-forward network to predict the next character at each
position. It is trained with cross-entropy loss, then generates text one
character at a time by sampling from its predictions.

## Model Architecture

The model is a compact, decoder-style character language model.
Character IDs are converted into token embeddings and combined with
learned positional embeddings. The representations pass through a
pre-normalized causal self-attention layer, then a residual connection.
A second normalization is followed by a feed-forward network, another
residual connection, and a linear projection that produces logits over
the character vocabulary. The causal mask prevents each position from
attending to future characters, which preserves the next-character
prediction setup used during training and generation.

<img width="3020" height="1448" alt="architecture" src="https://github.com/user-attachments/assets/f5129bdb-e014-4dd6-bdf7-939d5296803d" />

*Architecture flowchart : add the exported diagram at
`architecture.png`.*

### Main classes in the notebook

#### `DataPrep`

`DataPrep` prepares the character vocabulary and encoded data. It builds
a sorted list of unique characters, adds `?` as a fallback for unknown
characters, and creates the `stoi` (character-to-index) and `itos`
(index-to-character) mappings. Its `encode()` and `decode()` methods
convert between text and integer IDs. `split_data()` divides the text
sequentially into an 80% training portion and a 20% test/validation
portion, then returns encoded PyTorch tensors. The class also stores the
experiment settings: `embed_dim = 32`, `seq_len = 32`, and
`batch_size = 64`.

#### `cLLM` --- the language model

`cLLM` is the main `torch.nn.Module`. It combines token embeddings and
learned positional embeddings, applies LayerNorm before attention, adds
the attention output through a residual connection, applies another
LayerNorm before the feed-forward network, and adds a second residual
connection. A final linear layer maps each position's hidden
representation to logits for every character in the vocabulary.

The class also contains the training and inference methods.
`train_step()` computes cross-entropy loss, backpropagates it, and
updates the parameters with SGD (`lr=0.01`, `momentum=0.9`).
`validate()` calculates loss without gradient updates. `generate()`
repeatedly predicts a next-character distribution and samples from it
with `torch.multinomial`, appending each sampled character to the
context.

#### `Attention`

`Attention` implements **single-head causal self-attention**. Three
learned linear layers project the input into queries (`Q`), keys (`K`),
and values (`V`). It calculates the query-key dot products, scales them
by the square root of `embed_dim`, masks future positions with negative
infinity, applies softmax to obtain attention weights, and uses those
weights to combine the value vectors.

The causal mask is created with an upper-triangular matrix, so a
character position cannot use information from later positions. This is
essential for training the model to predict the next character without
leaking the target sequence's future information.

#### `FeedForward`

`FeedForward` applies a small multilayer perceptron independently to
each sequence position. Its hidden layer expands the representation from
`embed_dim` to `4 * embed_dim`, applies GELU, and projects it back to
`embed_dim`. In the current notebook, this class is the feed-forward/MLP
component used by `cLLM`.

**Class naming note:** The notebook defines `FeedForward`, but it does
not define a separate class named `MLP`. The feed-forward network itself
is implemented as an `nn.Sequential` pipeline inside `FeedForward`.

### Why this architecture?

I chose a small, explicit implementation so each component could be
inspected and tested independently. Character-level modelling keeps the
vocabulary compact and makes the next-character prediction objective
easy to observe. A causal attention mechanism reflects the
autoregressive behaviour of GPT-style models, while positional
embeddings provide sequence order and the feed-forward network
transforms each position's representation. Layer normalization and
residual connections help structure the processing flow. This design
keeps the implementation small enough to understand before moving to
larger models or pretrained LLM applications.

## Data Preparation

The notebook uses the Tiny Shakespeare text corpus. Each unique
character is assigned an integer ID through `stoi`, with `itos`
providing the reverse mapping.

The encoded text is split sequentially: the first 80% is used for
training and the remaining 20% for testing/validation. The training loop
uses `unfold()` to create overlapping windows of `seq_len + 1`
characters. Each window is split into an input sequence and a target
sequence shifted by one character.

For example:

``` text
Input:  "To be or not to b"
Target: "o be or not to be"
```

This means each input position is trained to predict the next character.
The vocabulary size in the recorded experiments was approximately **65
characters**; the current `DataPrep` implementation adds an extra
unknown-character token (`?`), so the exact size depends on the
characters present in the loaded dataset.

## Training

The notebook trains the model as a next-character predictor using
PyTorch.

  Setting                       Current notebook configuration
  ----------------------------- -----------------------------------
  Loss function                 Cross-entropy
  Optimizer                     SGD
  Learning rate                 `0.01`
  Momentum                      `0.9`
  Batch size                    `64`
  Embedding dimension           `32`
  Context length                `32`
  Attention                     Single-head causal self-attention
  Epochs in the training cell   `20`

The training loop creates overlapping input/target windows, shuffles the
window indices for each epoch, and trains in batches. For each batch,
the model produces logits for every position. The logits are permuted
into the shape expected by cross-entropy loss and compared with the
shifted targets. Backpropagation computes gradients and SGD updates the
model parameters. Validation loss is then calculated without gradient
updates, and both loss values are recorded for plotting.

Earlier experiments also used `embed_dim = 16` and `seq_len = 16`; those
results are summarized below.

## Experiments and Observations

The following are representative results recorded during development.
These were learning experiments rather than controlled benchmarks, so
differences between runs should not be treated as definitive
comparisons.

  -----------------------------------------------------------------------
  Experiment              Configuration           Recorded result
  ----------------------- ----------------------- -----------------------
  Embedding and attention `embed_dim=16`,         Training loss decreased
  development             `seq_len=16`, batch     to roughly `2.33`;
                          size `64`, up to 30     validation loss ended
                          epochs                  around `2.38`.

  Wider embeddings        `embed_dim=32`,         Training loss:
                          `seq_len=16`, batch     approximately `2.2735`;
                          size `64`, 20 epochs    validation loss:
                                                  approximately `2.3297`.

  Longer context          `embed_dim=32`,         Training loss:
                          `seq_len=32`, batch     approximately `2.3249`;
                          size `64`, 20 epochs    validation loss:
                                                  approximately `2.3803`.
  -----------------------------------------------------------------------

### What I observed

-   The decreasing loss showed that the model learned useful
    character-level patterns.
-   The run with `embed_dim=32` and `seq_len=16` recorded a lower
    validation loss than the longer-context run.
-   Increasing the context length from 16 to 32 did not improve the
    result in that particular experiment.
-   The experiments were not run with a fully controlled benchmark
    setup, so they do not establish that one configuration is always
    better.
-   Lower loss did not automatically produce coherent Shakespeare-like
    text. A small character-level model can learn local spelling and
    punctuation patterns without learning long-range structure.

## Text Generation

Generation is autoregressive: the model receives a context, predicts a
distribution over the next character, selects a character, appends it to
the context, and repeats. The `generate()` method crops the input to the
model's maximum context length before each prediction.

The first implementation used greedy selection with `argmax`. This often
produced repetitive text because the most likely character was selected
at every step. Switching to `torch.multinomial` sampling made the output
more varied, although it remained mostly incoherent. Sampling can
increase variation, but it cannot compensate for limitations in model
capacity, context length, or training.

## What I Learned

This project helped me connect the individual parts of a language model
into one working training pipeline. I learned how to encode text
numerically, prepare next-character targets, work with embedding and
attention tensor shapes, apply causal masking, calculate cross-entropy
loss, track training and validation performance, and generate text
autoregressively. The experiments also showed me why a falling loss is
useful but not sufficient evidence of high-quality generation.

## Current Limitations

-   Character-level tokenization is simple to inspect but less efficient
    than subword tokenization.
-   The model is small and uses a short context window.
-   The recorded training runs were limited and intended for learning.
-   Generated text became more varied with sampling but was not
    consistently coherent.
-   Random seeds and a standardized evaluation setup were not
    consistently recorded.
-   The notebook downloads the dataset and reads it from a local path,
    so the data-loading paths may need adjustment for another
    environment.

## Possible Next Steps

-   Make experiments reproducible with fixed seeds and saved
    configurations.
-   Plot training and validation loss across epochs.
-   Compare settings using controlled runs and the same training budget.
-   Experiment with temperature and top-k sampling.
-   Extend the architecture gradually after validating the current
    implementation.
-   Build a separate applied GenAI project using a pretrained model,
    RAG, API integration, evaluation, and deployment.

## Notebook and Setup

The implementation currently lives in a **Jupyter notebook**
(`LLM_from_scratch.ipynb`); there is no standalone Python model module
or `train.py` entry point.

1.  Clone the repository.
2.  Create and activate a Python virtual environment.
3.  Install PyTorch, scikit-learn, requests, and Jupyter.
4.  Open the notebook and run its cells in order.

Example environment setup:

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch scikit-learn requests jupyter matplotlib
jupyter notebook
```

The notebook downloads Tiny Shakespeare using `requests` and writes it
to `data-input.txt`, then reads the text from `/content/data-input.txt`.
That absolute path is specific to a Colab-style environment; update it
to match the location where you run the notebook.

## References

-   Andrej Karpathy, **Neural Networks: Zero to Hero**:
    https://github.com/karpathy/nn-zero-to-hero
-   Andrej Karpathy, **Let's build GPT: from scratch, in code, spelled
    out**: https://www.youtube.com/watch?v=kCc8FmEb1nY
-   Andrej Karpathy, **build-nanogpt**:
    https://github.com/karpathy/build-nanogpt
-   Tiny Shakespeare dataset:
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
