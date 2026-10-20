# bob — a full GPT, minus the training

A complete decoder-only transformer language model: real architecture, real
initialization, real inference. Weights are **random** — nothing has been
trained, by agreement. 🎲

## Layout

```
data/       # drop a plain-text corpus here (e.g. corpus.txt) if you ever train
models/     # checkpoints land here: gpt.pt + gpt.tokenizer.json
scripts/
  model.py          # GPT: causal attention w/ KV cache, MLP blocks, weight tying,
                    # GPT-2-style init, AdamW grouping, save/load
  tokenizer.py      # character-level tokenizer (train/save/load, zero downloads)
  generate.py       # sampling with temperature + top-k, KV-cached
  train.py          # complete training loop (warmup + cosine LR, grad clip,
                    # periodic eval) — exists, deliberately never run here
  verify_env.py     # honest environment + CUDA report + model smoke test
.venv/       # Python 3.14, PyTorch (CUDA build), transformers, datasets,
             # tokenizers, numpy, matplotlib, tqdm
```

## GPU reality check

This machine has **no Nvidia GPU** (only Intel UHD integrated graphics), so
`torch.cuda.is_available()` returns `False` and everything runs on CPU. The
CUDA build of PyTorch is installed as requested — move this directory to a
machine with an Nvidia GPU and the same code will use it with no reinstall.

Verify at any time:

```sh
.venv/bin/python scripts/verify_env.py
```

## Try the untrained model

```sh
.venv/bin/python scripts/generate.py --prompt "To be" --max-new-tokens 200 --seed 7
```

Expect gibberish — that is what an untrained distribution looks like.

## If you ever break the agreement and train

```sh
echo "your corpus text here" > data/corpus.txt
.venv/bin/python scripts/train.py --data data/corpus.txt --max-iters 5000
.venv/bin/python scripts/generate.py --checkpoint models/gpt.pt
```
