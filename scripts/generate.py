"""Sample from the model. Untrained weights = poetic gibberish; that's the demo.

Examples:
  .venv/bin/python scripts/generate.py                      # fresh random weights
  .venv/bin/python scripts/generate.py --prompt "To be" --max-new-tokens 200
  .venv/bin/python scripts/generate.py --checkpoint models/gpt.pt --temperature 0.8 --top-k 40
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Generate text from the GPT model")
    ap.add_argument("--checkpoint", type=str, default=None, help="path to a saved .pt (omit for random weights)")
    ap.add_argument("--prompt", type=str, default="\n", help="seed text")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.checkpoint:
        model, ckpt = GPT.load_checkpoint(args.checkpoint, device=device)
        tok = CharTokenizer.load(Path(args.checkpoint).with_suffix(".tokenizer.json"))
        print(f"Loaded checkpoint {args.checkpoint} ({model.num_params() / 1e6:.1f}M params)")
    else:
        # No checkpoint? Then this is the untrained experience, in full.
        tok = CharTokenizer.train_from_text(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:'\"-\n"
        )
        cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=256, n_layer=6, n_head=6, n_embd=384)
        model = GPT(cfg).to(device)
        print(f"No checkpoint given — using freshly initialized random weights "
              f"({model.num_params() / 1e6:.1f}M params). Output will be nonsense. 🎲")

    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx, args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, seed=args.seed,
    )
    print("\n" + tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
