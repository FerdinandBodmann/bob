//! Tool registry. `defs()` declares what the model sees, `run()` executes it.
//!
//! Adding a tool = one entry in `defs()`, one match arm in `run()`, one module.

pub mod fetch;
pub mod htmltext;
pub mod search;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use crate::provider::ToolDef;

/// Shared state for tool implementations.
#[derive(Clone)]
pub struct Ctx {
    pub http: reqwest::Client,
    /// Default character budget for a fetched page.
    pub fetch_limit: usize,
}

pub const USER_AGENT: &str =
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36 bobs/0.1 (+https://localhost)";

pub fn defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "fetch".into(),
            description:
                "Fetch an http(s) URL and return its readable text. HTML is stripped to plain \
                 text (navigation, scripts and styles removed). Use for docs, repos, articles, \
                 API responses. Returns [status], final URL after redirects, then the content."
                    .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute URL, including http:// or https://"
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Optional character cap for the returned text (default 24000, max 200000)"
                    }
                },
                "required": ["url"]
            }),
        },
        ToolDef {
            name: "search".into(),
            description:
                "Search the web. Returns up to N results as title, URL and snippet. Use before \
                 fetching when you do not know which URL to read."
                    .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "limit": { "type": "integer", "description": "Max results, 1-20 (default 8)" }
                },
                "required": ["query"]
            }),
        },
    ]
}

/// Execute a tool call. Errors are returned to the model as text (wrapped in
/// `<tool_error>`) rather than propagated — a failed fetch should not kill the
/// turn, the model can retry or change strategy.
pub async fn run(name: &str, args: &Value, ctx: &Ctx) -> String {
    let outcome: Result<String> = match name {
        "fetch" => fetch::run(args, ctx).await,
        "search" => search::run(args, ctx).await,
        other => Err(anyhow!("unknown tool '{other}'")),
    };
    match outcome {
        Ok(out) => out,
        Err(e) => format!("<tool_error>{e:#}</tool_error>"),
    }
}

/// Human-readable one-liner for the `→` progress line.
pub fn summarize(name: &str, args: &Value) -> String {
    let arg = |key: &str| {
        args.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| args.get(key).map(|v| v.to_string()).unwrap_or_default())
    };
    match name {
        "fetch" => format!("fetch {}", arg("url")),
        "search" => format!("search {:?}", arg("query")),
        other => format!("{other} {args}"),
    }
}
