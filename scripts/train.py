"""A complete, ready-to-run training loop — present so the model is "full".

Per the standing agreement: this file exists but is NOT executed here.
To actually train someday:

  .venv/bin/python scripts/train.py --data data/corpus.txt --max-iters 5000
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def lr_schedule(it, max_iters, warmup, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    ratio = (it - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/corpus.txt")
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--out", type=str, default="models/gpt.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    text = Path(args.data).read_text(encoding="utf-8")
    tok = CharTokenizer.train_from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    tok.save(Path(args.out).with_suffix(".tokenizer.json"))

    if len(data) < 2:
        sys.exit(f"corpus is too small: {len(data)} tokens after tokenization — "
                 f"put an actual text in {args.data}")
    if len(data) < args.block_size + 2:
        new_bs = max(4, len(data) - 2)
        print(f"warning: corpus has only {len(data)} tokens but block_size={args.block_size}; "
              f"shrinking block_size to {new_bs}. This is a smoke test, not real training.")
        args.block_size = new_bs

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                    n_layer=6, n_head=6, n_embd=384, dropout=0.1)
    model = GPT(cfg).to(device)
    opt = model.configure_optimizers(args.weight_decay, args.lr, betas=(0.9, 0.95))
    print(f"{model.num_params() / 1e6:.1f}M parameters on {device}")

    t0 = time.perf_counter()
    for it in range(args.max_iters + 1):
        lr = lr_schedule(it, args.max_iters, args.warmup, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters:
            model.eval()
            losses = []
            with torch.no_grad():
                for _ in range(10):
                    x, y = get_batch(data, args.block_size, args.batch_size, device)
                    _, loss, _ = model(x, y)
                    losses.append(loss.item())
            print(f"iter {it:5d} | eval loss {sum(losses) / len(losses):.4f} | lr {lr:.2e}")

        x, y = get_batch(data, args.block_size, args.batch_size, device)
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 100 == 0:
            dt = time.perf_counter() - t0
            print(f"iter {it:5d} | train loss {loss.item():.4f} | {dt:.1f}s")
            t0 = time.perf_counter()

    model.save_checkpoint(args.out, iter=args.max_iters)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
