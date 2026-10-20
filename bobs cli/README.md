# bobs

A small agentic AI CLI with live internet access. Written in Rust, no runtime
dependencies beyond your API key.

The model can **search the web** and **fetch pages** while it answers, so it can
read current docs, repos, release notes, and Stack Overflow instead of guessing.

## Build

```sh
cargo build --release
# binary at target/release/bobs, or:
cargo install --path .
```

## Configure

bobs picks up credentials automatically, in this order:

1. `--provider` / `BOBS_PROVIDER`
2. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `XAI_API_KEY`, `GEMINI_API_KEY`
3. keys already stored by OpenCode in `~/.local/share/opencode/auth.json`

```sh
export ANTHROPIC_API_KEY=sk-ant-…
bobs doctor          # show what bobs resolved + connectivity check
```

| provider    | default endpoint                                    | default model          |
| ----------- | --------------------------------------------------- | ---------------------- |
| `anthropic` | `https://api.anthropic.com/v1`                      | `claude-sonnet-4-5`    |
| `openai`    | `https://api.openai.com/v1`                         | `gpt-4o-mini`          |
| `openrouter`| `https://openrouter.ai/api/v1`                      | `openai/gpt-4o-mini`   |
| `xai`       | `https://api.x.ai/v1`                               | `grok-4`               |
| `gemini`    | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.5-flash` |
| `local`     | `http://localhost:11434/v1` (Ollama / llama.cpp)    | `llama3.2`             |

Anything speaking OpenAI's `/chat/completions` works:
`bobs --provider openai --base-url https://your-endpoint/v1 --model your-model`.

## Use

```sh
bobs                              # interactive chat
bobs ask "what changed in rustc since 1.85?"   # one-shot
echo "summarize this issue" | bobs ask         # piped input
bobs fetch https://docs.rs/reqwest             # the same tool the model uses
bobs search "rust reqwest 0.13 release notes"
bobs tools                                     # tool schemas
bobs doctor                                    # config + connectivity
```

Useful flags: `--model`, `--no-tools` (plain chat), `--max-steps`, `--fetch-limit`,
`--system "…"`, `--temperature`.

### REPL commands

```
/help   /tools   /sys <text>   /reset   /fetch <url>   /search <query>   /exit
```

## Layout

```
src/
  main.rs            subcommand dispatch
  cli.rs             clap definitions
  config.rs          provider/key/model resolution
  agent.rs           the tool loop: model → calls → results → model
  repl.rs            terminal loop (rustyline + streaming print)
  provider/
    mod.rs           Msg/ToolDef/Step/Event types, SSE line buffer
    anthropic.rs     Messages API, native tool_use, streaming
    openai.rs        OpenAI-compatible chat/completions, streaming
  tools/
    mod.rs           registry + dispatcher
    fetch.rs         HTTP fetch, HTML → text, truncation
    htmltext.rs      dependency-free HTML stripper
    search.rs        DuckDuckGo Lite search, no API key
  ui.rs              ANSI styling
```

## Extending

* **New tool** — add a `ToolDef` in `tools/mod.rs::defs()`, a match arm in
  `tools/mod.rs::run()`, and implement it in its own module. The model picks it
  up automatically.
* **New provider host** (a vendor, not a dialect) — one `Spec` arm in
  `config.rs::provider_spec()`.
* **New provider dialect** — implement `complete()` in `provider/`, add a
  `Kind` variant and a match arm in `provider/mod.rs::Client::complete`.
* **Search API** (Brave, Serper, Tavily, Exa) — replace `tools/search.rs::run()`
  and keep the `Title / URL / Snippet` output shape.
* **Local transcript storage** — `agent.rs` owns the `Vec<Msg>`; persist it as
  JSON and reload on startup.

Run the parsing tests with `cargo test` (HTML stripper and the search parser are
covered by fixtures).
