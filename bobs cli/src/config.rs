//! Provider + runtime configuration.
//!
//! Resolution order (highest wins):
//!   1. CLI flags (`--provider`, `--model`, `--base-url`, …)
//!   2. `BOBS_*` environment variables
//!   3. Plain API-key env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …)
//!   4. Credentials already stored by OpenCode in `~/.local/share/opencode/auth.json`

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::cli::GlobalArgs;
use crate::provider::Kind;

pub const DEFAULT_SYSTEM: &str = "\
You are bobs, a terse and capable terminal assistant with live internet access.
Use the fetch and search tools whenever the answer depends on current information,
documentation, or something you cannot know for certain. Prefer primary sources
(official docs, repos, standards) over aggregators. Cite URLs you actually fetched.
Answer in plain text suitable for a terminal: no markdown tables, short paragraphs,
bulleted lists only when they help. If a tool fails, say so and work around it.";

/// A fully resolved provider configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub provider: String,
    pub kind: Kind,
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub system: String,
    pub max_steps: usize,
    pub fetch_limit: usize,
    pub no_tools: bool,
}

impl Config {
    pub fn resolve(args: &GlobalArgs) -> Result<Config> {
        let provider_name = args
            .provider
            .clone()
            .or_else(|| std::env::var("BOBS_PROVIDER").ok())
            .or_else(detect_provider)
            .context(no_key_help())?;

        let spec = provider_spec(&provider_name)?;

        let env_suffix = provider_name.to_uppercase().replace('-', "_");
        let api_key = args
            .api_key
            .clone()
            .or_else(|| nonempty(std::env::var(format!("BOBS_{env_suffix}_API_KEY")).ok()))
            .or_else(|| nonempty(std::env::var(spec.key_env).ok()))
            .or_else(|| nonempty(std::env::var("BOBS_API_KEY").ok()))
            .or_else(|| opencode_key(&provider_name))
            .unwrap_or_default();

        if api_key.is_empty() && spec.requires_key {
            bail!(
                "no API key found for provider '{provider_name}'.\n\
                 Set {} (or BOBS_API_KEY), or run `bobs doctor` to see what bobs can see.",
                spec.key_env
            );
        }

        let base_url = args
            .base_url
            .clone()
            .or_else(|| std::env::var("BOBS_BASE_URL").ok())
            .unwrap_or_else(|| spec.base_url.to_string());

        let model = args
            .model
            .clone()
            .or_else(|| std::env::var("BOBS_MODEL").ok())
            .unwrap_or_else(|| spec.default_model.to_string());

        let system = args
            .system
            .clone()
            .or_else(|| std::env::var("BOBS_SYSTEM").ok())
            .unwrap_or_else(|| DEFAULT_SYSTEM.to_string());

        Ok(Config {
            provider: provider_name,
            kind: spec.kind,
            api_key,
            base_url,
            model,
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            system,
            max_steps: args.max_steps.max(1),
            fetch_limit: args.fetch_limit.max(500),
            no_tools: args.no_tools,
        })
    }
}

struct Spec {
    kind: Kind,
    base_url: &'static str,
    default_model: &'static str,
    key_env: &'static str,
    requires_key: bool,
}

/// Known providers. Everything except Anthropic speaks the OpenAI
/// `/chat/completions` dialect, so adding a new one is usually one line here.
fn provider_spec(name: &str) -> Result<Spec> {
    let spec = match name {
        "anthropic" => Spec {
            kind: Kind::Anthropic,
            base_url: "https://api.anthropic.com/v1",
            default_model: "claude-sonnet-4-5",
            key_env: "ANTHROPIC_API_KEY",
            requires_key: true,
        },
        "openai" => Spec {
            kind: Kind::OpenAi,
            base_url: "https://api.openai.com/v1",
            default_model: "gpt-4o-mini",
            key_env: "OPENAI_API_KEY",
            requires_key: true,
        },
        "openrouter" => Spec {
            kind: Kind::OpenAi,
            base_url: "https://openrouter.ai/api/v1",
            default_model: "openai/gpt-4o-mini",
            key_env: "OPENROUTER_API_KEY",
            requires_key: true,
        },
        "xai" => Spec {
            kind: Kind::OpenAi,
            base_url: "https://api.x.ai/v1",
            default_model: "grok-4",
            key_env: "XAI_API_KEY",
            requires_key: true,
        },
        // Gemini's OpenAI-compatible surface.
        "gemini" | "google" => Spec {
            kind: Kind::OpenAi,
            base_url: "https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model: "gemini-2.5-flash",
            key_env: "GEMINI_API_KEY",
            requires_key: true,
        },
        // Local or self-hosted servers: llama.cpp, vLLM, LM Studio, Ollama.
        "local" | "ollama" => Spec {
            kind: Kind::OpenAi,
            base_url: "http://localhost:11434/v1",
            default_model: "llama3.2",
            key_env: "BOBS_API_KEY",
            requires_key: false,
        },
        other => bail!(
            "unknown provider '{other}'. Known: anthropic, openai, openrouter, xai, gemini, local.\n\
             For anything else use --provider openai --base-url https://your-endpoint/v1"
        ),
    };
    Ok(spec)
}

fn detect_provider() -> Option<String> {
    // Env keys first, then whatever OpenCode has already authenticated.
    for (env, name) in [
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
        ("OPENROUTER_API_KEY", "openrouter"),
        ("XAI_API_KEY", "xai"),
        ("GEMINI_API_KEY", "gemini"),
        ("GOOGLE_API_KEY", "gemini"),
    ] {
        if std::env::var(env).map(|v| !v.is_empty()).unwrap_or(false) {
            return Some(name.to_string());
        }
    }
    for name in ["anthropic", "openrouter", "xai", "google", "ollama-cloud"] {
        if opencode_key(name).is_some() {
            return Some(if name == "google" { "gemini".into() } else { name.into() });
        }
    }
    None
}

fn no_key_help() -> &'static str {
    "no AI provider configured.\n\
     Set a key (ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY / …),\n\
     or pick one explicitly: bobs --provider anthropic, or run `bobs doctor`."
}

/// Pull a credential out of OpenCode's auth store, if present.
fn opencode_key(provider: &str) -> Option<String> {
    let path = opencode_auth_path()?;
    let text = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let entry = value.get(provider)?;
    entry.get("key").and_then(|k| k.as_str()).map(|s| s.to_string())
}

fn opencode_auth_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".local/share/opencode/auth.json");
    p.exists().then_some(p)
}

fn nonempty(v: Option<String>) -> Option<String> {
    v.filter(|s| !s.is_empty())
}
