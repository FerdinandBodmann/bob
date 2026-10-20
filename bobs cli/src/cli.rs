//! Command-line surface (clap).

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "bobs",
    version,
    about = "bobs — a small agentic AI CLI with live internet access",
    long_about = None,
    arg_required_else_help = false
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    #[command(flatten)]
    pub global: GlobalArgs,
}

#[derive(Args, Debug, Clone)]
pub struct GlobalArgs {
    /// Provider: anthropic | openai | openrouter | xai | gemini | local
    #[arg(long, global = true)]
    pub provider: Option<String>,

    /// Model id (defaults per provider)
    #[arg(long, global = true)]
    pub model: Option<String>,

    /// Override the API base URL (for proxies, vLLM, llama.cpp, …)
    #[arg(long, global = true)]
    pub base_url: Option<String>,

    /// API key (prefer env vars — this shows up in your shell history)
    #[arg(long, global = true, hide = true)]
    pub api_key: Option<String>,

    /// Max tokens per response
    #[arg(long, global = true, default_value_t = 4096)]
    pub max_tokens: u32,

    /// Sampling temperature
    #[arg(long, global = true)]
    pub temperature: Option<f32>,

    /// Replace the default system prompt
    #[arg(long, global = true, short = 's')]
    pub system: Option<String>,

    /// Disable tools — plain chat, no internet access
    #[arg(long, global = true)]
    pub no_tools: bool,

    /// Max tool rounds per user prompt
    #[arg(long, global = true, default_value_t = 12)]
    pub max_steps: usize,

    /// Character budget for a fetched page
    #[arg(long, global = true, default_value_t = 24_000)]
    pub fetch_limit: usize,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Interactive chat (default)
    Chat {
        /// Optional first prompt; without one you drop into the REPL
        #[arg(trailing_var_arg = true)]
        prompt: Vec<String>,
    },

    /// One-shot: print the answer and exit (reads piped stdin too)
    Ask {
        /// The prompt; omit to read it from stdin
        #[arg(trailing_var_arg = true)]
        prompt: Vec<String>,
    },

    /// Fetch a URL and print readable text (same tool the model uses)
    Fetch {
        url: String,
        /// Character cap
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Web search (same tool the model uses)
    Search {
        #[arg(trailing_var_arg = true)]
        query: Vec<String>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
    },

    /// List the tools the model can call
    Tools,

    /// Show resolved configuration, credentials and connectivity
    Doctor,
}
