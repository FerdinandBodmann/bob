"""Environment verification: versions, CUDA detection, folders, smoke test.

Run: .venv/bin/python scripts/verify_env.py
"""

import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    import torch
except ImportError:
    sys.exit("torch is not installed yet — install it first.")

print(f"Python      : {platform.python_version()} ({sys.executable})")
print(f"Platform    : {platform.platform()}")
print(f"PyTorch     : {torch.__version__}")

# --- CUDA detection (the honest part) ---------------------------------------
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version   : {torch.version.cuda}")
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"GPU memory     : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    device = torch.device("cuda")
else:
    reason = (
        "torch is a CPU-only build" if torch.version.cuda is None
        else "CUDA build present, but no usable Nvidia GPU/driver on this machine"
    )
    print(f"Reason         : {reason}")
    print("Falling back to CPU — everything below still works.")
    device = torch.device("cpu")

# --- required packages -------------------------------------------------------
print("\nPackages:")
missing = []
for name in ["torch", "transformers", "datasets", "tokenizers", "numpy", "matplotlib", "tqdm"]:
    try:
        mod = __import__(name)
        print(f"  {name:<12} {getattr(mod, '__version__', '?')}")
    except ImportError:
        missing.append(name)
        print(f"  {name:<12} MISSING")
if missing:
    sys.exit(f"\nMissing packages: {', '.join(missing)}")

# --- project structure -------------------------------------------------------
print("\nProject structure:")
for folder in ["data", "models", "scripts"]:
    p = ROOT / folder
    status = "ok" if p.is_dir() else "MISSING — run: mkdir -p data models scripts"
    print(f"  {folder}/  {status}")
    if not p.is_dir():
        sys.exit(1)

# --- model smoke test on the target device -----------------------------------
sys.path.insert(0, str(ROOT / "scripts"))
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

print(f"\nModel smoke test on '{device}':")
tok = CharTokenizer.train_from_text("hello world")
cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=2, n_embd=64)
model = GPT(cfg).to(device)
print(f"  parameters : {model.num_params() / 1e6:.2f}M (char-level demo config)")

x = torch.randint(0, tok.vocab_size, (2, 32), device=device)
y = torch.randint(0, tok.vocab_size, (2, 32), device=device)
t0 = time.perf_counter()
logits, loss, _ = model(x, y)
loss.backward()
elapsed = (time.perf_counter() - t0) * 1e3
print(f"  fwd+bwd    : {elapsed:.1f} ms  (loss={loss.item():.2f}, untrained random weights)")

out = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 20, seed=0)
print(f"  generate   : ok, got {out.shape[1] - 1} random tokens")

if device.type == "cuda":
    torch.cuda.synchronize()
print("\nAll good. Environment is ready (model is NOT trained — that's intentional).")
