//! Provider abstraction.
//!
//! Two dialects cover almost everything worth talking to:
//!   * [`Kind::Anthropic`]  — Messages API with native `tool_use` / `tool_result`
//!   * [`Kind::OpenAi`]     — OpenAI-compatible `/chat/completions` (OpenRouter,
//!     xAI, Gemini, vLLM, llama.cpp, Ollama, …)
//!
//! Add a dialect by implementing `complete()` for a new `Kind` and adding it to
//! the match in [`Client::complete`]. Add a *host* by adding a `Spec` in
//! [`crate::config`] — no provider code needed.

pub mod anthropic;
pub mod openai;

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Kind {
    Anthropic,
    OpenAi,
}

/// A tool the model is allowed to call. `parameters` is a JSON Schema object.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
}

/// The internal conversation representation, translated per provider.
#[derive(Debug, Clone, Serialize)]
pub enum Msg {
    System(String),
    User(String),
    Assistant {
        text: Option<String>,
        calls: Vec<ToolCall>,
    },
    Tool {
        id: String,
        name: String,
        output: String,
    },
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

/// One provider round-trip: text the model wrote plus the tools it invoked.
#[derive(Debug, Clone, Default)]
pub struct Step {
    pub text: String,
    pub calls: Vec<ToolCall>,
    pub usage: Usage,
}

/// Progress callbacks fired while streaming. Wire these to the terminal.
#[derive(Debug, Clone)]
pub enum Event {
    Text(String),
    ToolStart { name: String, args: Value },
}

pub type Sink<'a> = &'a mut dyn FnMut(Event);

#[derive(Clone)]
pub struct Client {
    pub kind: Kind,
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub http: reqwest::Client,
}

impl Client {
    pub fn describe(&self) -> String {
        let dialect = match self.kind {
            Kind::Anthropic => "anthropic",
            Kind::OpenAi => "openai-compat",
        };
        format!("{} · {} · {dialect}", self.base_url, self.model)
    }

    pub async fn complete(
        &self,
        msgs: &[Msg],
        tools: &[ToolDef],
        sink: Sink<'_>,
    ) -> Result<Step> {
        match self.kind {
            Kind::Anthropic => anthropic::complete(self, msgs, tools, sink).await,
            Kind::OpenAi => openai::complete(self, msgs, tools, sink).await,
        }
    }
}

/// Incremental line splitter for SSE payloads: chunks arrive at arbitrary byte
/// boundaries and events end with `\n\n`.
#[derive(Default)]
pub struct LineBuffer {
    buf: Vec<u8>,
}

impl LineBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a chunk, get back every complete line it finished.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            let text = String::from_utf8_lossy(&line[..line.len() - 1]).trim().to_string();
            if !text.is_empty() {
                out.push(text);
            }
        }
        out
    }
}

/// Pull the `data:` payload out of an SSE line, if any.
pub fn sse_data(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("data:")?;
    let rest = rest.trim();
    (!rest.is_empty()).then_some(rest)
}

/// Turn a non-2xx response body into a readable message.
pub fn api_error(status: u16, body: &str, where_: &str) -> anyhow::Error {
    let detail = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message")
                .or_else(|| v.pointer("/message"))
                .or_else(|| v.pointer("/error/0/message"))
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| body.chars().take(600).collect::<String>().trim().to_string());
    anyhow::anyhow!("{where_} returned {status}: {detail}")
}
